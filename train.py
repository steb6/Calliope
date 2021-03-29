import os
import torch
from datetime import datetime
from tqdm.auto import tqdm
from config import config, remote
from iterate_dataset import SongIterator
from optimizer import CTOpt
from loss_computer import SimpleLossCompute, compute_accuracy, LabelSmoothing
from create_bar_dataset import NoteRepresentationManager
import wandb
from compressive_transformer import CompressiveEncoder, CompressiveDecoder
from compress_latents import LatentCompressor, LatentDecompressor
from logger import Logger
from discriminator import Discriminator
from loss_computer import calc_gradient_penalty
from config import set_freer_gpu
import time
from utilities import get_prior
from test import Tester
import dill


class Trainer:
    def __init__(self):
        self.logger = None
        self.tester = None
        self.latent = None
        self.save_path = None
        self.epoch = 0
        self.step = 0
        self.loss_computer = None
        self.tf_prob = 0
        # Models
        self.encoder = None
        self.latent_compressor = None
        self.latent_decompressor = None
        self.decoder = None
        if config["train"]["aae"]:
            self.discriminator = None
        # Optimizers
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        if config["train"]["aae"]:
            self.disc_optimizer = None
            self.gen_optimizer = None
            self.train_discriminator_not_generator = True
            self.disc_losses = []
            self.gen_losses = []
            self.disc_loss_init = None
            self.gen_loss_init = None
            self.beta = -0.1  # so it become 0 at first iteration
            self.reg_optimizer = None

    def test_losses(self, loss, e_attn_losses, d_attn_losses):
        losses = [loss, e_attn_losses, d_attn_losses]
        names = ["loss", "e_att_losses", "d_attn_losses"]
        for ls, name in zip(losses, names):
            print("********************** Optimized by " + name)
            self.encoder_optimizer.optimizer.zero_grad(set_to_none=True)
            self.decoder_optimizer.zero_grad(set_to_none=True)
            ls.backward(retain_graph=True)
            for model in [self.encoder, self.latent_compressor, self.decoder]:  # removed latent compressor
                for module_name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        print(module_name)
        self.encoder_optimizer.optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)
        (losses[0] + losses[1] + losses[2]).backward(retain_graph=True)
        print("********************** NOT OPTIMIZED BY NOTHING")
        for model in [self.encoder, self.latent_compressor, self.decoder]:  # removed latent compressor
            for module_name, parameter in model.named_parameters():
                if parameter.grad is None:
                    print(module_name)

    def run_mb(self, batch):
        # SETUP VARIABLES
        srcs, trgs, src_masks, trg_masks, trg_ys = batch
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 2)
        trgs = torch.LongTensor(trgs.long()).to(config["train"]["device"]).transpose(0, 2)  # invert batch and bars
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"]).transpose(0, 2)
        trg_masks = torch.BoolTensor(trg_masks).to(config["train"]["device"]).transpose(0, 2)
        trg_ys = torch.LongTensor(trg_ys.long()).to(config["train"]["device"]).transpose(0, 2)
        latent = None

        ############
        # ENCODING #
        ############
        latents = []
        for src, src_mask in zip(srcs, src_masks):
            latent = self.encoder(src, src_mask)
            latents.append(latent)

        ############
        # COMPRESS #
        ############
        if config["train"]["compress_latents"]:
            latent = self.latent_compressor(latents)  # in: 3, 4, 200, 256, out: 3, 256

        self.latent = latent.detach().cpu().numpy()

        if config["train"]["compress_latents"]:
            latents = self.latent_decompressor(latent)  # in 3, 256, out: 3, 4, 200, 256

        ############
        # DECODING #
        ############
        # Scheduled sampling for transformer
        if config["train"]["scheduled_sampling"]:
            for _ in range(1):  # K
                self.tf_prob = 0.5

                predicted = []
                for trg, trg_mask, latent in zip(trgs, trg_masks, latents):
                    out = self.decoder(trg, trg_mask, latent.transpose(0, 1))
                    predicted.append(torch.max(out, dim=-1).indices)

                # add sos at beginning and cut last token
                predicted = torch.stack(predicted)
                sos = torch.full_like(predicted, config["tokens"]["sos"])[..., :1].to(predicted.device)
                predicted = torch.cat((sos, predicted), dim=-1)[..., :-1]
                # create mixed trg
                mixed_prob = torch.rand(trgs.shape, dtype=torch.float32).to(trgs.device)
                mixed_prob = mixed_prob < self.tf_prob
                trgs = trgs.where(mixed_prob, predicted)

        outs = []
        for trg, trg_mask, src_mask, latent in zip(trgs, trg_masks, src_masks, latents):
            out = self.decoder(trg, trg_mask, latent.transpose(0, 1))
            outs.append(out)

        # Format results
        outs = torch.stack(outs, dim=0)

        # Compute loss and accuracy
        loss, loss_items = self.loss_computer(outs, trg_ys)
        predicted = torch.max(outs, dim=-1).indices
        accuracy = compute_accuracy(predicted, trg_ys, config["tokens"]["pad"])
        losses = (loss.item(), accuracy, 0, 0, *loss_items)

        # LOG IMAGES
        if self.encoder.training and config["train"]["log_images"] and \
                self.step % config["train"]["after_steps_log_images"] == 0:

            # ENCODER SELF
            drums_encoder_attn = []
            for layer in self.encoder.drums_encoder.layers:
                instrument_attn = []
                for head in layer.self_attn.attn[0]:
                    instrument_attn.append(head)
                drums_encoder_attn.append(instrument_attn)

            bass_encoder_attn = []
            for layer in self.encoder.bass_encoder.layers:
                instrument_attn = []
                for head in layer.self_attn.attn[0]:
                    instrument_attn.append(head)
                bass_encoder_attn.append(instrument_attn)

            guitar_encoder_attn = []
            for layer in self.encoder.guitar_encoder.layers:
                instrument_attn = []
                for head in layer.self_attn.attn[0]:
                    instrument_attn.append(head)
                guitar_encoder_attn.append(instrument_attn)

            strings_encoder_attn = []
            for layer in self.encoder.strings_encoder.layers:
                instrument_attn = []
                for head in layer.self_attn.attn[0]:
                    instrument_attn.append(head)
                strings_encoder_attn.append(instrument_attn)

            enc_attention = [drums_encoder_attn, bass_encoder_attn, guitar_encoder_attn, strings_encoder_attn]

            # DECODER SELF
            drums_decoder_attn = []
            for layer in self.decoder.drums_decoder.layers:
                instrument_attn = []
                for head in layer.self_attn.attn[0]:
                    instrument_attn.append(head)
                drums_decoder_attn.append(instrument_attn)

            bass_decoder_attn = []
            for layer in self.decoder.bass_decoder.layers:
                instrument_attn = []
                for head in layer.self_attn.attn[0]:
                    instrument_attn.append(head)
                bass_decoder_attn.append(instrument_attn)

            guitar_decoder_attn = []
            for layer in self.decoder.guitar_decoder.layers:
                instrument_attn = []
                for head in layer.self_attn.attn[0]:
                    instrument_attn.append(head)
                guitar_decoder_attn.append(instrument_attn)

            strings_decoder_attn = []
            for layer in self.decoder.strings_decoder.layers:
                instrument_attn = []
                for head in layer.self_attn.attn[0]:
                    instrument_attn.append(head)
                strings_decoder_attn.append(instrument_attn)

            dec_attention = [drums_decoder_attn, bass_decoder_attn, guitar_decoder_attn, strings_decoder_attn]
            # DECODER SRC
            drums_src_attn = []
            for layer in self.decoder.drums_decoder.layers:
                instrument_attn = []
                for head in layer.src_attn.attn[0]:
                    instrument_attn.append(head)
                drums_src_attn.append(instrument_attn)

            bass_src_attn = []
            for layer in self.decoder.bass_decoder.layers:
                instrument_attn = []
                for head in layer.src_attn.attn[0]:
                    instrument_attn.append(head)
                bass_src_attn.append(instrument_attn)

            guitar_src_attn = []
            for layer in self.decoder.guitar_decoder.layers:
                instrument_attn = []
                for head in layer.src_attn.attn[0]:
                    instrument_attn.append(head)
                guitar_src_attn.append(instrument_attn)

            strings_src_attn = []
            for layer in self.decoder.strings_decoder.layers:
                instrument_attn = []
                for head in layer.src_attn.attn[0]:
                    instrument_attn.append(head)
                strings_src_attn.append(instrument_attn)

            src_attention = [drums_src_attn, bass_src_attn, guitar_src_attn, strings_src_attn]

            print("Logging images...")
            self.logger.log_latent(self.latent)
            self.logger.log_attn_heatmap(enc_attention, dec_attention, src_attention)
            self.logger.log_examples(srcs, trgs)

        ####################
        # UPDATE GENERATOR #
        ####################
        if self.encoder.training:
            self.encoder.zero_grad()
            self.latent_compressor.zero_grad()
            self.latent_decompressor.zero_grad()
            self.decoder.zero_grad()

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)
            # torch.nn.utils.clip_grad_norm_(self.latent_compressor.parameters(), 0.1)
            # torch.nn.utils.clip_grad_norm_(self.latent_decompressor.parameters(), 0.1)
            # torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        if config["train"]["aae"] and self.encoder.training:

            if self.step % config["train"]["increase_beta_every"] == 0 and self.beta < config["train"]["max_beta"]:
                self.beta += 0.1

            if self.beta > 0:
                # To suppress warnings
                D_real = 0
                D_fake = 0
                loss_critic = 0

                ########################
                # UPDATE DISCRIMINATOR #
                ########################
                for p in self.encoder.parameters():
                    p.requires_grad = False
                for p in self.latent_compressor.parameters():
                    p.requires_grad = False
                for p in self.discriminator.parameters():
                    p.requires_grad = True

                for _ in range(config["train"]["critic_iterations"]):
                    prior = get_prior((config["train"]["batch_size"], config["model"]["d_model"]))  # autograd is intern
                    D_real = self.discriminator(prior).reshape(-1)

                    latents = []
                    for src, src_mask in zip(srcs, src_masks):
                        latent = self.encoder(src, src_mask)
                        latents.append(latent)
                    latent = self.latent_compressor(latents)
                    D_fake = self.discriminator(latent).reshape(-1)

                    gradient_penalty = calc_gradient_penalty(self.discriminator, prior.data, latent.data)

                    loss_critic = (
                            torch.mean(D_fake) - torch.mean(D_real) + config["train"]["lambda"] * gradient_penalty
                    )
                    loss_critic = loss_critic * self.beta

                    self.discriminator.zero_grad()
                    loss_critic.backward(retain_graph=True)

                    self.disc_optimizer.step(lr=self.encoder_optimizer.lr)

                ####################
                # UPDATE GENERATOR #
                ####################
                for p in self.encoder.parameters():
                    p.requires_grad = True
                for p in self.latent_compressor.parameters():
                    p.requires_grad = True
                for p in self.discriminator.parameters():
                    p.requires_grad = False  # to avoid computation

                latents = []
                for src, src_mask in zip(srcs, src_masks):
                    latent = self.encoder(src, src_mask)
                    latents.append(latent)
                latent = self.latent_compressor(latents)

                G = self.discriminator(latent).reshape(-1)

                loss_gen = -torch.mean(G)
                loss_gen = loss_gen * self.beta

                self.encoder.zero_grad()
                self.latent_compressor.zero_grad()
                loss_gen.backward()

                self.gen_optimizer.step(lr=self.encoder_optimizer.lr)

                losses += (D_real.mean().cpu().data.numpy(), D_fake.mean().cpu().data.numpy(),
                           G.mean().cpu().data.numpy(), loss_critic.cpu().data.numpy(), loss_gen.cpu().data.numpy(),
                           D_real.mean().cpu().data.numpy() - D_fake.mean().cpu().data.numpy())

        return losses

    def train(self):
        # Create checkpoint folder
        if not os.path.exists(config["paths"]["checkpoints"]):
            os.makedirs(config["paths"]["checkpoints"])
        timestamp = str(datetime.now())
        timestamp = timestamp[:timestamp.index('.')]
        timestamp = timestamp.replace(' ', '_').replace(':', '-')
        self.save_path = config["paths"]["checkpoints"] + os.sep + timestamp
        os.mkdir(self.save_path)

        # Create models
        self.encoder = CompressiveEncoder().to(config["train"]["device"])
        self.latent_compressor = LatentCompressor(config["model"]["d_model"]).to(config["train"]["device"])
        self.latent_decompressor = LatentDecompressor(config["model"]["d_model"]).to(config["train"]["device"])
        self.decoder = CompressiveDecoder().to(config["train"]["device"])
        if config["train"]["aae"]:
            self.discriminator = Discriminator(config["model"]["d_model"],
                                               config["model"]["discriminator_dropout"]).to(config["train"]["device"])

        # Create optimizers
        self.encoder_optimizer = CTOpt(torch.optim.Adam([{"params": self.encoder.parameters()},
                                                         {"params": self.latent_compressor.parameters()}], lr=0,
                                                        betas=(0., 0.9)),
                                       config["train"]["warmup_steps"],
                                       (config["train"]["lr_min"], config["train"]["lr_max"]),
                                       config["train"]["decay_steps"], config["train"]["minimum_lr"]
                                       )
        self.decoder_optimizer = CTOpt(torch.optim.Adam([{"params": self.latent_decompressor.parameters()},
                                                         {"params": self.decoder.parameters()}], lr=0,
                                                        betas=(0., 0.9)),
                                       config["train"]["warmup_steps"],
                                       (config["train"]["lr_min"], config["train"]["lr_max"]),
                                       config["train"]["decay_steps"], config["train"]["minimum_lr"])

        if config["train"]["aae"]:
            self.disc_optimizer = CTOpt(torch.optim.Adam([{"params": self.discriminator.parameters()}], lr=0,
                                                         betas=(0., 0.9)),
                                        config["train"]["warmup_steps"],
                                        (config["train"]["lr_min"], config["train"]["lr_max"]),
                                        config["train"]["decay_steps"], config["train"]["minimum_lr"]
                                        )
            self.gen_optimizer = CTOpt(torch.optim.Adam([{"params": self.encoder.parameters()},
                                                         {"params": self.latent_compressor.parameters()}], lr=0,
                                                        betas=(0., 0.9)),
                                       config["train"]["warmup_steps"],
                                       (config["train"]["lr_min"], config["train"]["lr_max"]),
                                       config["train"]["decay_steps"], config["train"]["minimum_lr"]
                                       )
        # Loss computer
        criterion = LabelSmoothing(size=config["tokens"]["vocab_size"],
                                   padding_idx=config["tokens"]["pad"],
                                   smoothing=config["train"]["label_smoothing"],
                                   device=config["train"]["device"])
        criterion.to(config["train"]["device"])
        self.loss_computer = SimpleLossCompute(criterion)

        # Load dataset
        dataset = SongIterator(dataset_path=config["paths"]["dataset"],
                               test_size=config["train"]["test_size"],
                               batch_size=config["train"]["batch_size"],
                               n_workers=config["train"]["n_workers"])
        tr_loader, ts_loader = dataset.get_loaders()

        # Init WANDB
        self.logger = Logger()
        wandb.login()
        wandb.init(project="MusAE", config=config, name="r_" + timestamp if remote else "l_" + timestamp)
        wandb.watch(self.encoder, log_freq=1000, log="all")
        wandb.watch(self.latent_compressor, log_freq=1000, log="all")
        wandb.watch(self.latent_decompressor, log_freq=1000, log="all")
        wandb.watch(self.decoder, log_freq=1000, log="all")
        if config["train"]["aae"]:
            wandb.watch(self.discriminator, log_freq=1000, log="all")

        # Print info about training
        time.sleep(1.)  # sleep for one second to let the machine connect to wandb
        if config["train"]["verbose"]:
            cmem_range = config["model"]["cmem_len"] * config["model"]["cmem_ratio"]
            max_range = config["model"]["layers"] * (cmem_range + config["model"]["mem_len"])
            given = config["data"]["bars"] * config["model"]["seq_len"]
            assert given <= max_range, "Given {} as input to model with max range {}".format(given, max_range)
            print("Giving", given, "as input to model with a maximum range of", max_range)
            print("Giving", len(tr_loader), "training samples and", len(ts_loader), "test samples")
            print("Final set has size", len(dataset.final_set))
            print("Model has", config["model"]["layers"], "layers")
            print("Batch size is", config["train"]["batch_size"])
            print("d_model is", config["model"]["d_model"])
            if config["train"]["aae"]:
                print("Imposing prior distribution on latents")
                print("Starting training aae after", config["train"]["train_aae_after_steps"])
                print("lambda:", config["train"]["lambda"], ", critic iterations:",
                      config["train"]["critic_iterations"])
            else:
                print("NOT imposing prior distribution on latents")
            if config["train"]["log_images"]:
                print("Logging images")
            else:
                print("NOT logging images")
            if config["train"]["make_songs"]:
                print("making songs")
            else:
                print("NOT making songs")
            if config["train"]["do_eval"]:
                print("doing evaluation")
            else:
                print("NOT DOING evaluation")
            if config["train"]["scheduled_sampling"]:
                print("Using scheduled sampling")
            else:
                print("NOT using scheduled sampling")
            if config["train"]["compress_latents"]:
                print("Compressing latents")
            else:
                print("NOT compressing latents")
            if config["train"]["use_src_mask"]:
                print("using src mask")
            else:
                print("NOT using src mask")
            if config["train"]["use_rel_pos"]:
                print("Using relative positional encoding")
            else:
                print("NOT using relative positional encoding")

        # Setup train
        self.encoder.train()
        self.latent_compressor.train()
        self.latent_decompressor.train()
        self.decoder.train()
        if config["train"]["aae"]:
            self.discriminator.train()
        desc = "Train epoch " + str(self.epoch) + ", mb " + str(0)
        train_progress = tqdm(total=config["train"]["steps_before_eval"], position=0, leave=True, desc=desc)
        self.step = 0  # -1 to do eval in first step
        first_batch = None

        # Main loop
        for self.epoch in range(config["train"]["n_epochs"]):  # for each epoch
            for song_it, batch in enumerate(tr_loader):  # for each song

                #########
                # TRAIN #
                #########
                if first_batch is None:  # if training reconstruct from train, if eval reconstruct from eval
                    first_batch = batch
                second_batch = batch
                tr_losses = self.run_mb(batch)

                self.logger.log_losses(tr_losses, self.encoder.training)
                self.logger.log_stuff(self.encoder_optimizer.lr,
                                      self.latent,
                                      self.disc_optimizer.lr if config["train"]["aae"] else None,
                                      self.gen_optimizer.lr if config["train"]["aae"] else None,
                                      self.beta if config["train"]["aae"] else None,
                                      get_prior(self.latent.shape) if config["train"]["aae"] else None,
                                      self.tf_prob)
                if self.step == 0:
                    print("Latent shape is:", self.latent.shape)
                train_progress.update()

                ########
                # EVAL #
                ########
                if self.step % config["train"]["steps_before_eval"] == 0 and config["train"]["do_eval"]:
                    print("Evaluation")
                    train_progress.close()
                    ts_losses = []

                    self.encoder.eval()
                    self.latent_compressor.eval()
                    self.latent_decompressor.eval()
                    self.decoder.eval()

                    if config["train"]["aae"]:
                        self.discriminator.eval()
                    desc = "Eval epoch " + str(self.epoch) + ", mb " + str(song_it)

                    # Compute validation score
                    first_batch = None
                    for test in tqdm(ts_loader, position=0, leave=True, desc=desc):  # remember test losses
                        if first_batch is None:
                            first_batch = test
                        second_batch = test
                        with torch.no_grad():
                            ts_loss = self.run_mb(test)
                        ts_losses.append(ts_loss)
                    final = ()  # average losses
                    for i in range(len(ts_losses[0])):  # for each loss value
                        aux = []
                        for loss in ts_losses:  # for each computed loss
                            aux.append(loss[i])
                        avg = sum(aux) / len(aux)
                        final = final + (avg,)
                    self.logger.log_losses(final, self.encoder.training)

                    # eval end
                    self.encoder.train()
                    self.latent_compressor.train()
                    self.latent_decompressor.train()
                    self.decoder.train()
                    if config["train"]["aae"]:
                        self.discriminator.train()
                    desc = "Train epoch " + str(self.epoch) + ", mb " + str(song_it)
                    train_progress = tqdm(total=config["train"]["steps_before_eval"], position=0, leave=True,
                                          desc=desc)

                ##############
                # SAVE MODEL #
                ##############
                if (self.step % config["train"]["after_steps_save_model"]) == 0:
                    full_path = self.save_path + os.sep + str(self.step)
                    os.makedirs(full_path)
                    print("Saving last model in " + full_path + ", DO NOT INTERRUPT")
                    torch.save(self.encoder, os.path.join(full_path, "encoder.pt"), pickle_module=dill)
                    torch.save(self.latent_compressor, os.path.join(full_path,
                                                                    "latent_compressor.pt"), pickle_module=dill)
                    torch.save(self.latent_decompressor, os.path.join(full_path,
                                                                      "latent_decompressor.pt"), pickle_module=dill)
                    torch.save(self.decoder, os.path.join(full_path, "decoder.pt"), pickle_module=dill)
                    if config["train"]["aae"]:
                        torch.save(self.discriminator, os.path.join(full_path, "discriminator.pt"), pickle_module=dill)
                    print("Model saved")

                ########
                # TEST #
                ########
                if (self.step % config["train"]["after_steps_make_songs"]) == 0 and config["train"]["make_songs"]:
                    print("Making songs")
                    self.encoder.eval()
                    self.latent_compressor.eval()
                    self.latent_decompressor.eval()
                    self.decoder.eval()

                    self.tester = Tester(self.encoder, self.latent_compressor, self.latent_decompressor, self.decoder)

                    # RECONSTRUCTION
                    note_manager = NoteRepresentationManager()
                    to_reconstruct = first_batch
                    with torch.no_grad():
                        original, reconstructed, limited, acc = self.tester.reconstruct(to_reconstruct, note_manager)
                    prefix = "epoch_" + str(self.epoch) + "_mb_" + str(song_it)
                    self.logger.log_songs(os.path.join(wandb.run.dir, prefix),
                                          [original, reconstructed, limited],
                                          ["original", "reconstructed", "limited"],
                                          "validation reconstruction example")
                    self.logger.log_reconstruction_accuracy(acc)

                    if config["train"]["aae"]:
                        # GENERATION
                        with torch.no_grad():
                            generated, limited = self.tester.generate(note_manager)  # generation
                        self.logger.log_songs(os.path.join(wandb.run.dir, prefix),
                                              [generated, limited],
                                              ["generated", "limited"],
                                              "generated")

                        # INTERPOLATION
                        with torch.no_grad():
                            first, interpolation, limited, second = self.tester.interpolation(note_manager, first_batch,
                                                                                              second_batch)

                        self.logger.log_songs(os.path.join(wandb.run.dir, prefix),
                                              [first, interpolation, limited, second],
                                              ["first", "interpolation", "limited", "second"],
                                              "interpolation")
                    # end test
                    self.encoder.train()
                    self.latent_compressor.train()
                    self.latent_decompressor.train()
                    self.decoder.train()

                self.step += 1


if __name__ == "__main__":
    set_freer_gpu()
    trainer = Trainer()
    trainer.train()
