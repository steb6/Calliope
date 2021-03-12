import os
import torch
from datetime import datetime
from tqdm.auto import tqdm
from config import config, remote
from iterate_dataset import SongIterator
from optimizer import CTOpt
from loss_computer import SimpleLossCompute, compute_accuracy, LabelSmoothing
from create_bar_dataset import NoteRepresentationManager  # TODO check
import glob
import wandb
from compressive_transformer import CompressiveEncoder, CompressiveDecoder
from compress_latents import LatentCompressor
import numpy as np
from logger import Logger
from utilities import get_memories, create_trg_mask, midi_to_wav
from discriminator import Discriminator
from torch.autograd import Variable
from loss_computer import calc_gradient_penalty
from config import set_freer_gpu
import time
from utilities import get_prior
from test import Tester
import random


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
            self.decoder.optimizer.zero_grad(set_to_none=True)
            ls.backward(retain_graph=True)
            for model in [self.encoder, self.decoder]:  # removed latent compressor
                for module_name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        print(module_name)
        self.encoder_optimizer.optimizer.zero_grad(set_to_none=True)
        self.decoder.optimizer.zero_grad(set_to_none=True)
        (losses[0] + losses[1] + losses[2]).backward(retain_graph=True)
        print("********************** NOT OPTIMIZED BY NOTHING")
        for model in [self.encoder, self.decoder]:  # removed latent compressor
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
        e_attn_losses = []
        d_attn_losses = []
        outs = []
        enc_self_weights = []
        dec_self_weights = []
        dec_src_weights = []
        latent = None
        e_mems, e_cmems, d_mems, d_cmems = get_memories()

        # Encode
        for src, src_mask in zip(srcs, src_masks):
            latent, e_mems, e_cmems, e_attn_loss, sw = self.encoder(src, src_mask, e_mems, e_cmems)
            # e_mems = e_mems.detach()
            # e_cmems = e_cmems.detach()
            enc_self_weights.append(sw)
            e_attn_losses.append(e_attn_loss)

        latent = self.latent_compressor(latent)
        self.latent = latent.detach().cpu().numpy()
        # dec_latent = latent.reshape(config["train"]["batch_size"], config["model"]["n_latents"],
        #                             config["model"]["d_model"])

        # TODO ##################################
        # TODO SCHEDULED SAMPLING FOR TRANSFORMER
        # TODO ##################################
        # dec_latent = latent
        # # first pass: compute decoder output
        # with torch.no_grad():
        #     for trg, src_mask, trg_mask in zip(trgs, src_masks, trg_masks):
        #         out, _, _, d_mems, d_cmems, _ = self.decoder(trg, trg_mask, src_mask, dec_latent, d_mems, d_cmems)
        #         # d_mems = d_mems.detach()
        #         # d_cmems = d_cmems.detach()
        #         outs.append(out)
        # outs = torch.stack(outs)
        #
        # # select best k tokens and their probabilities
        # top_k = torch.topk(outs, config["train"]["top_k_mixed_embeddings"], dim=-1)
        # outs = top_k.indices  # 8 1 200 5 4
        # prob = torch.exp(top_k.values)
        # # permute
        # # outs = outs.permute(0, 4, 1, 2, 3)
        # # prob = prob.permute(0, 4, 1, 2, 3)
        # # transpose right of one dimension
        # sos = torch.full_like(outs, config["tokens"]["sos"], device=outs.device)[:, :, :, :1, :]
        # outs = torch.cat((sos, outs), dim=-2)
        # outs = outs[:, :, :, :-1, :]
        # fill_prob = torch.full_like(prob, 0.1, device=outs.device)[:, :, :, :1, :]
        # prob = torch.cat((fill_prob, prob), dim=-2)
        # prob = prob[:, :, :, :-1, :]
        # # scale probabilities
        # scaled_prob = torch.zeros_like(prob)
        # for b, bar in enumerate(prob):
        #     for i, instrument in enumerate(bar):
        #         for ba, batch in enumerate(instrument):
        #             for t, token in enumerate(batch):
        #                 s = torch.sum(prob[b][i][ba][t])
        #                 for e, _ in enumerate(token):
        #                     scaled_prob[b][i][ba][t][e] = prob[b][i][ba][t][e] / s
        # # mix gold and predicted with given probability
        # self.tf_prob = max(config["train"]["min_tf_prob"],
        #                    config["train"]["max_tf_prob"] - self.step * config["train"]["tf_prob_step_reduction"])
        # mix = torch.zeros_like(outs)
        # for b, bar in enumerate(outs):
        #     for i, instrument in enumerate(bar):
        #         for ba, batch in enumerate(instrument):
        #             for t, token in enumerate(batch):
        #                 if random.random() < self.tf_prob:  # probability to do teacher forcing
        #                     mix[b][i][ba][t] = trgs[b][i][ba][t].unsqueeze(-1).expand_as(outs[b][i][ba][t])
        #                 else:
        #                     mix[b][i][ba][t] = outs[b][i][ba][t]
        # # second pass with mixed target
        # _, _, d_mems, d_cmems = get_memories()
        # outs = []
        # for m, p, src_mask, trg_mask in zip(mix, scaled_prob, src_masks, trg_masks):
        #     out, self_weight, src_weight, d_mems, d_cmems, d_attn_loss = self.decoder(m, trg_mask, src_mask,
        #                                                                               dec_latent,
        #                                                                               d_mems, d_cmems, emb_weights=p)
        #     # d_mems = d_mems.detach()
        #     # d_cmems = d_cmems.detach()
        #     d_attn_losses.append(d_attn_loss)
        #     outs.append(out)
        #     dec_self_weights.append(self_weight)
        #     dec_src_weights.append(src_weight)
        # TODO ##################################
        # TODO SCHEDULED SAMPLING FOR TRANSFORMER
        # TODO ##################################

        # TODO ##################################
        # TODO GREEDY DECODING
        # TODO ##################################
        _, _, d_mems, d_cmems = get_memories(n_batch=1)
        outs = []
        self.tf_prob = max(config["train"]["min_tf_prob"],
                           config["train"]["max_tf_prob"] - self.step * config["train"]["tf_prob_step_reduction"])
        n_batch, n_latents = latent.shape
        for i in range(len(srcs)):
            trg = np.full((4, 1, 1), config["tokens"]["sos"])
            trg = torch.LongTensor(trg).to(config["train"]["device"])
            k = None
            v = None
            bar_one_hot = torch.zeros((n_batch, n_latents), dtype=torch.float32, device=trg.device)
            m = latent.shape[1]//len(trgs)
            bar_one_hot[:, (i*m):((i+1)*m)] = 1.
            bar_one_hot = Variable(bar_one_hot)
            bar_latent = torch.cat((latent, bar_one_hot), dim=0)
            for j in range(config["model"]["seq_len"] - 1):  # for each token of each bar
                if random.random() < self.tf_prob:
                    trg = torch.cat((trg, trgs[i, :, :, j+1:j+2]), dim=-1)  # teacher forcing, add element j+1
                else:
                    trg_mask = create_trg_mask(trg.cpu().numpy())
                    out, _, _, _, _, _, _, _ = self.decoder(trg, trg_mask, bar_latent, k, v)  # TODO CAREFUL FAST
                    out = torch.max(out, dim=-1).indices
                    trg = torch.cat((trg, out[..., -1:]), dim=-1)
            trg_mask = create_trg_mask(trg.cpu().numpy())
            out, self_weight, src_weight, _, _, d_attn_loss, k, v = self.decoder(trg, trg_mask, bar_latent, k, v)
            dec_self_weights.append(self_weight.detach())
            dec_src_weights.append(src_weight.detach())
            d_attn_losses.append(d_attn_loss.detach())
            outs.append(out)
        # TODO ##################################
        # TODO END GREEDY DECODING
        # TODO ##################################
        outs = torch.stack(outs, dim=0)
        e_attn_losses = torch.stack(e_attn_losses).mean()
        d_attn_losses = torch.stack(d_attn_losses).mean()

        loss, loss_items = self.loss_computer(outs, trg_ys)

        # Compute accuracy
        predicted = torch.max(outs, dim=-1).indices
        accuracy = compute_accuracy(predicted, trg_ys, config["tokens"]["pad"])

        losses = (loss.item(), accuracy, e_attn_losses.item(), d_attn_losses.item(), *loss_items)

        # SOME TESTS
        if self.encoder.training and config["train"]["log_images"]:
            if self.step % config["train"]["after_steps_log_images"] == 0:
                print("Logging images...")
                self.logger.log_latent(self.latent)
                enc_self_weights = torch.stack(enc_self_weights)
                dec_self_weights = torch.stack(dec_self_weights)
                dec_src_weights = torch.stack(dec_src_weights)
                self.logger.log_attn_heatmap(enc_self_weights, dec_self_weights, dec_src_weights)
                self.logger.log_memories(e_mems, e_cmems, d_mems, d_cmems)
                self.logger.log_examples(srcs, trgs, outs, trg_ys)
            if self.step == 0 and config["train"]["test_losses"]:
                self.test_losses(loss, e_attn_losses, d_attn_losses)

        # OPTIMIZE
        if self.encoder.training:
            # Reconstruction
            self.encoder.zero_grad()
            self.latent_compressor.zero_grad()
            self.decoder.zero_grad()

            optimizing_losses = loss + e_attn_losses + d_attn_losses
            optimizing_losses.backward()

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.latent_compressor.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

        if config["train"]["aae"] and self.encoder.training:  # TODO adjust for evaluation

            if self.step % config["train"]["increase_beta_every"] == 0 and self.beta < config["train"]["max_beta"]:
                self.beta += 0.1

            if self.beta > 0:
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
                    prior = get_prior(latent.shape)  # autograd is intern
                    # prior = Variable(prior).to(config["train"]["device"])
                    D_real = self.discriminator(prior).reshape(-1)

                    e_mems, e_cmems, _, _ = get_memories()
                    for src, src_mask in zip(srcs, src_masks):
                        latent, e_mems, e_cmems, _, _ = self.encoder(src, src_mask, e_mems, e_cmems)
                        e_mems = e_mems.detach()
                        e_cmems = e_cmems.detach()
                    latent = self.latent_compressor(latent)
                    D_fake = self.discriminator(latent).reshape(-1)

                    gradient_penalty = calc_gradient_penalty(self.discriminator, prior.data,
                                                             latent.data)

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

                e_mems, e_cmems, _, _ = get_memories()
                for src, src_mask in zip(srcs, src_masks):
                    latent, e_mems, e_cmems, _, _ = self.encoder(src, src_mask, e_mems, e_cmems)
                    e_mems = e_mems.detach()
                    e_cmems = e_cmems.detach()
                latent = self.latent_compressor(latent)

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
        self.decoder = CompressiveDecoder().to(config["train"]["device"])
        if config["train"]["aae"]:
            self.discriminator = Discriminator(config["model"]["d_model"],
                                               config["model"]["discriminator_dropout"]).to(config["train"]["device"])

        # Create optimizers
        self.encoder_optimizer = CTOpt(torch.optim.Adam([{"params": self.encoder.parameters()},
                                                         {"params": self.latent_compressor.parameters()}], lr=0),
                                       config["train"]["warmup_steps"],
                                       (config["train"]["lr_min"], config["train"]["lr_max"]),
                                       config["train"]["decay_steps"], config["train"]["minimum_lr"]
                                       )
        self.decoder_optimizer = CTOpt(torch.optim.Adam([{"params": self.decoder.parameters()}], lr=0),
                                       config["train"]["warmup_steps"],
                                       (config["train"]["lr_min"], config["train"]["lr_max"]),
                                       config["train"]["decay_steps"], config["train"]["minimum_lr"])
        if config["train"]["aae"]:
            self.disc_optimizer = CTOpt(torch.optim.Adam([{"params": self.discriminator.parameters()}], lr=0),
                                        config["train"]["warmup_steps"],
                                        (config["train"]["lr_min"], config["train"]["lr_max"]),
                                        config["train"]["decay_steps"], config["train"]["minimum_lr"]
                                        )
            self.gen_optimizer = CTOpt(torch.optim.Adam([{"params": self.encoder.parameters()},
                                                         {"params": self.latent_compressor.parameters()}], lr=0),
                                       config["train"]["warmup_steps"],
                                       (config["train"]["lr_min"], config["train"]["lr_max"]),
                                       config["train"]["decay_steps"], config["train"]["minimum_lr"]
                                       )
        # Loss
        criterion = LabelSmoothing(size=config["tokens"]["vocab_size"],
                                   padding_idx=config["tokens"]["pad"],
                                   smoothing=config["train"]["label_smoothing"],
                                   device=config["train"]["device"])
        criterion.to(config["train"]["device"])
        self.loss_computer = SimpleLossCompute(criterion)

        # Dataset
        dataset = SongIterator(dataset_path=config["paths"]["dataset"],
                               test_size=config["train"]["test_size"],
                               batch_size=config["train"]["batch_size"],
                               n_workers=config["train"]["n_workers"])
        tr_loader, ts_loader = dataset.get_loaders()

        # Wandb
        self.logger = Logger()
        wandb.login()
        wandb.init(project="MusAE", config=config, name="r_" + timestamp if remote else "l_" + timestamp)
        wandb.watch(self.encoder)
        wandb.watch(self.latent_compressor)
        wandb.watch(self.decoder)
        if config["train"]["aae"]:
            wandb.watch(self.discriminator)

        # Print info about training
        time.sleep(1.)  # sleep for one second to let the machine connect to wandb
        if config["train"]["verbose"]:
            print("Bar size reduced to ", config["train"]["n_bars"])
            cmem_range = config["model"]["cmem_len"] * config["model"]["cmem_ratio"]
            max_range = config["model"]["layers"] * (cmem_range + config["model"]["mem_len"])
            given = config["train"]["n_bars"] * config["model"]["seq_len"]
            assert given <= max_range, "Given {} as input to model with max range {}".format(given, max_range)
            print("Giving ", given, " as input to model with a maximum range of ", max_range)
            print("Giving ", len(tr_loader), " training samples and ", len(ts_loader), " test samples")
            print("Giving ", config["data"]["truncated_bars"], " bars to a model with ",
                  config["model"]["layers"], " layers")
            print("Latent size:", config["model"]["n_latents"] * config["model"]["d_model"])
            if config["train"]["aae"]:
                print("Imposing prior distribution on latents")
                print("Starting training aae after ", config["train"]["train_aae_after_steps"])
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

        # Train
        self.encoder.train()
        self.latent_compressor.train()
        self.decoder.train()
        if config["train"]["aae"]:
            self.discriminator.train()
        desc = "Train epoch " + str(self.epoch) + ", mb " + str(0)
        train_progress = tqdm(total=config["train"]["steps_before_eval"], position=0, leave=True, desc=desc)
        self.step = 0  # -1 to do eval in first step
        first_batch = None  # TODO remove
        # main loop
        for self.epoch in range(config["train"]["n_epochs"]):  # for each epoch
            for song_it, batch in enumerate(tr_loader):  # for each song

                #########
                # TRAIN #
                #########
                if first_batch is None:  # TODO remove
                    first_batch = batch  # TODO remove
                tr_losses = self.run_mb(batch)

                self.logger.log_losses(tr_losses, self.encoder.training)
                self.logger.log_stuff(self.encoder_optimizer.lr,
                                      self.latent,
                                      self.disc_optimizer.lr if config["train"]["aae"] else None,
                                      self.gen_optimizer.lr if config["train"]["aae"] else None,
                                      self.beta if config["train"]["aae"] else None,
                                      get_prior(self.latent.shape) if config["train"]["aae"] else None,
                                      self.tf_prob)
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
                    self.decoder.eval()

                    if config["train"]["aae"]:
                        self.discriminator.eval()
                    desc = "Eval epoch " + str(self.epoch) + ", mb " + str(song_it)

                    # Compute validation score
                    # first = None  TODO put it back
                    for test in tqdm(ts_loader, position=0, leave=True, desc=desc):  # remember test losses
                        # if first is None:  TODO put it back
                        #     first = test  TODO put it back
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
                    torch.save(self.encoder, os.path.join(full_path, "encoder.pt"))
                    torch.save(self.latent_compressor, os.path.join(full_path,
                                                                    "latent_compressor.pt"))
                    torch.save(self.decoder, os.path.join(full_path, "decoder.pt"))
                    print("Model saved")

                ########
                # TEST #
                ########
                if (self.step % config["train"]["after_steps_make_songs"]) == 0 and config["train"]["make_songs"]:
                    print("Making songs")
                    self.encoder.eval()
                    self.latent_compressor.eval()
                    self.decoder.eval()

                    self.tester = Tester(self.encoder, self.latent_compressor, self.decoder)

                    # RECONSTRUCTION
                    note_manager = NoteRepresentationManager()
                    test = batch  # TODO remove
                    to_reconstruct = test
                    with torch.no_grad():
                        original, reconstructed, limited = self.tester.reconstruct(to_reconstruct, note_manager)
                    prefix = "epoch_" + str(self.epoch) + "_mb_" + str(song_it)
                    self.logger.log_songs(os.path.join(wandb.run.dir, prefix),
                                          [original, reconstructed, limited],
                                          ["original", "reconstructed", "limited"],
                                          "validation reconstruction example")

                    if config["train"]["aae"]:
                        # GENERATION
                        with torch.no_grad():
                            generated, limited = self.tester.generate(note_manager)  # generation
                        self.logger.log_songs(os.path.join(wandb.run.dir, prefix),
                                              [generated, limited],
                                              ["generated", "limited"],
                                              "generated")

                        # INTERPOLATION
                        second = test
                        with torch.no_grad():
                            first, interpolation, limited, second = self.tester.interpolation(note_manager, first_batch,
                                                                                              second)

                        self.logger.log_songs(os.path.join(wandb.run.dir, prefix),
                                              [first, interpolation, limited, second],
                                              ["first", "interpolation", "limited", "second"],
                                              "interpolation")
                    # end test
                    self.encoder.train()
                    self.latent_compressor.train()
                    self.decoder.train()

                self.step += 1


if __name__ == "__main__":
    set_freer_gpu()
    trainer = Trainer()
    trainer.train()
