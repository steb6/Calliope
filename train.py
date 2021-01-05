import os
import torch
from datetime import datetime
from tqdm.auto import tqdm
from config import config, remote
from iterate_dataset import SongIterator
from optimizer import CTOpt
from label_smoother import LabelSmoothing
from loss_computer import SimpleLossCompute
from torch.autograd import Variable
from create_dataset import NoteRepresentationManager
import glob
import wandb
from midi_converter import midi_to_wav
from discriminator import Discriminator
from compressive_transformer import CompressiveEncoder, CompressiveDecoder, LatentsCompressor, LatentsDecompressor
import numpy as np
from PIL import Image
from torch.nn import functional as f


class Trainer:
    def __init__(self, save_path=None, model_name="checkpoint", plot_name="plot", settings=config):
        self.save_path = save_path
        self.model_name = model_name
        self.plot_name = plot_name
        self.settings = settings
        self.epoch = 0
        self.step = 0
        self.loss_computer = None
        # Models
        self.encoder = None
        self.compressor = None
        self.decompressor = None
        self.decoder = None
        if config["train"]["aae"]:
            self.discriminator = None
        # Optimizers
        self.model_optimizer = None
        self.compressor_optimizer = None
        if config["train"]["aae"]:
            self.discriminator_optimizer = None
            self.generator_optimizer = None

    def test_losses(self, loss, e_attn_losses, e_ae_losses, d_attn_losses, d_ae_losses, recon_loss):
        losses = [loss, e_attn_losses, e_ae_losses, d_attn_losses, d_ae_losses, recon_loss]
        names = ["loss", "e_att_losses", "e_ae_losses", "d_attn_losses", "d_ae_losses", "latents_reconstruction_loss"]
        for ls, name in zip(losses, names):
            print("********************** Optimized by " + name)
            self.model_optimizer.optimizer.zero_grad(set_to_none=True)
            ls.backward(retain_graph=True)
            for model in [self.encoder, self.compressor, self.decompressor, self.decoder]:
                for module_name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        print(module_name)

    @staticmethod
    def log_attn_images(mem_weights, self_weights, lat_weights):
        mem_img = mem_weights.detach().cpu().numpy()
        mem_formatted = (mem_img * 255 / np.max(mem_img)).astype('uint8')
        mem_img = Image.fromarray(mem_formatted)

        self_img = self_weights.detach().cpu().numpy()
        self_formatted = (self_img * 255 / np.max(self_img)).astype('uint8')
        self_img = Image.fromarray(self_formatted)

        lat_img = lat_weights.detach().cpu().numpy()
        lat_formatted = (lat_img * 255 / np.max(lat_img)).astype('uint8')
        lat_img = Image.fromarray(lat_formatted)

        pad = np.array([255 * (i % 2) for i in range(config["model"]["seq_len"] * 9)]).reshape(-1, 9).astype(np.uint8)
        pad_img = Image.fromarray(pad)
        # CONCATENATE IMAGES
        images = [mem_img, pad_img, self_img, pad_img, lat_img]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        wandb.log({"examples": [wandb.Image(new_im, caption="Encoder memory attention weights, decoder self"
                                                            " attention weights and decoder source attention"
                                                            " weights")]})

    @staticmethod
    def log_examples(e_in, d_in, pred, exp, lat, z, r_lat):  # TODO log latents
        enc_input = e_in.permute(1, 2, 0, 3)[0].reshape(4, -1).detach().cpu().numpy()
        dec_input = d_in.permute(1, 2, 0, 3)[0].reshape(4, -1).detach().cpu().numpy()
        predicted = torch.max(pred, dim=-2).indices.permute(0, 2, 1)[0].reshape(4, -1).detach().cpu().numpy()
        expected = exp.permute(1, 2, 0, 3)[0].reshape(4, -1).detach().cpu().numpy()
        latent = lat.permute(2, 1, 0, 3, 4)[0].reshape(4, -1, config["model"]["d_model"]).detach().cpu().numpy()
        columns = ["Encoder Input: " + str(enc_input.shape),
                   "Decoder Input: " + str(dec_input.shape),
                   "Predicted: " + str(predicted.shape),
                   "Expected: " + str(expected.shape),
                   "Original latents: " + str(latent.shape)]
        inputs = (enc_input, dec_input, predicted, expected, latent)
        if z is not None:
            zeta = z[0].detach().cpu().numpy()
            inputs = inputs + (zeta,)
            columns.append("Z: " + str(z.shape))
        if r_lat is not None:
            recon_l = r_lat.permute(2, 1, 0, 3, 4)[0].reshape(4, -1, config["model"]["d_model"]).detach().cpu().numpy()
            inputs = inputs + (recon_l,)
            columns.append("Reconstructed latents: " + str(recon_l.shape))
        table = wandb.Table(columns=columns)
        table.add_data(*inputs)
        wandb.log({"out": table})

    @staticmethod
    def get_memories():
        a = 4
        b = config["model"]["layers"]
        c = config["train"]["batch_size"]
        e = config["model"]["d_model"]
        device = config["train"]["device"]
        e_mems = torch.zeros(a, b, c, 0, e, dtype=torch.float32, device=device)
        e_cmems = torch.zeros(a, b, c, 0, e, dtype=torch.float32, device=device)
        d_mems = torch.zeros(a, b, c, 0, e, dtype=torch.float32, device=device)
        d_cmems = torch.zeros(a, b, c, 0, e, dtype=torch.float32, device=device)
        return e_mems, e_cmems, d_mems, d_cmems

    @staticmethod
    def create_trg_mask(trg):
        trg_mask = np.full(trg.shape + (trg.shape[-1],), True)
        for b in range(config["train"]["batch_size"]):
            for i in range(4):
                line_mask = trg[b][i] != config["tokens"]["pad"]
                pad_mask = np.matmul(line_mask[:, np.newaxis], line_mask[np.newaxis, :])
                subsequent_mask = np.expand_dims(np.tril(np.ones((trg.shape[-1], trg.shape[-1]))), (0, 1))
                subsequent_mask = subsequent_mask.astype(np.bool)
                trg_mask[b][i] = pad_mask & subsequent_mask
        trg_mask = torch.BoolTensor(trg_mask).to(config["train"]["device"])
        return trg_mask

    def greedy_decoding(self, latent, src_mask, d_mems, d_cmems):
        trg = np.full((config["train"]["batch_size"], 4, 1), config["tokens"]["sos"])
        for _ in range(config["model"]["seq_len"] - 1):
            trg_mask = self.create_trg_mask(trg)
            trg = torch.LongTensor(trg).to(config["train"]["device"])
            out, _, _, _, _ = self.decoder(trg, latent, src_mask, trg_mask, d_mems, d_cmems)
            out = torch.max(out, dim=-2).indices
            out = out.transpose(1, 2)
            trg = torch.cat((trg, out[..., -1:]), dim=-1)
            trg = trg.detach().cpu().numpy()
        trg_mask = self.create_trg_mask(trg)
        trg = torch.LongTensor(trg).to(config["train"]["device"])
        return trg, trg_mask

    def run_mb(self, batch):
        # Train reconstruction
        srcs, trgs, src_masks, trg_masks, trg_ys = batch
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 1)
        trgs = torch.LongTensor(trgs.long()).to(config["train"]["device"]).transpose(0, 1)  # invert batch and seq
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"]).transpose(0, 1)
        trg_masks = torch.BoolTensor(trg_masks).to(config["train"]["device"]).transpose(0, 1)
        trg_ys = torch.LongTensor(trg_ys.long()).to(config["train"]["device"]).transpose(0, 1)
        e_mems, e_cmems, d_mems, d_cmems = self.get_memories()
        e_attn_losses = []
        e_ae_losses = []
        d_attn_losses = []
        d_ae_losses = []
        latents = []
        outs = []
        aws = []
        # Encode
        for src, src_mask in zip(srcs, src_masks):
            latent, e_mems, e_cmems, e_attn_loss, e_ae_loss, aw = self.encoder(src, src_mask, e_mems, e_cmems)
            aws.append(aw)
            e_attn_losses.append(e_attn_loss)
            e_ae_losses.append(e_ae_loss)
            latents.append(latent)
        # Compress and decompress latents  # TODO detach all attn weights
        # PAD AWS
        length = max([s.shape[-1] for s in aws])
        for count, aw in enumerate(aws):
            aws[count] = f.pad(aw, (length - aw.shape[-1], 0))
        aws = torch.mean(torch.stack(aws, dim=0), dim=0)
        reconstructed_latents = None
        z = None
        # END PAD AWS
        # act = torch.nn.Tanh()
        # original_latents = act(original_latents)
        original_latents = torch.stack(latents, dim=2)  # 1 x 4 x 10 x 301 x 32
        # TODO REMOVE TEST
        # import copy
        # original_latents_copy = copy.deepcopy(original_latents.data)
        z = self.compressor(original_latents)  # 1 x z_dim
        reconstructed_latents = self.decompressor(z)
        loss = torch.nn.MSELoss()
        latents_reconstruction_loss = loss(reconstructed_latents, original_latents)
        reconstructed_latents = reconstructed_latents.transpose(0, 2)
        original_latents = original_latents.transpose(0, 2)
        # Decode
        latents_weight = []
        self_weights = []
        for latent, src_mask, trg, trg_mask in zip(original_latents, src_masks, trgs, trg_masks):
            if not self.decoder.training:  # do not use teacher forcing
                trg, trg_mask = self.greedy_decoding(latent, src_mask, d_mems, d_cmems)
            # out, d_mems, d_cmems, d_attn_loss, d_ae_loss
            out, latent_weight, self_weight = self.decoder(trg, latent, src_mask, trg_mask)  # , d_mems, d_cmems)
            # d_attn_losses.append(d_attn_loss)
            # d_ae_losses.append(d_ae_loss)
            outs.append(out)
            latents_weight.append(latent_weight)
            self_weights.append(self_weight)
        latents_weight = torch.mean(torch.stack(latents_weight, dim=0), dim=0)
        self_weights = torch.mean(torch.stack(self_weights, dim=0), dim=0)
        # LOG ATTENTION IMAGES
        if self.step % config["train"]["after_mb_log_attn_img"] == 0:
            self.log_attn_images(aws, self_weights, latents_weight)

        outs = torch.stack(outs, dim=1)
        outs = outs.reshape(config["train"]["batch_size"], -1, config["tokens"]["vocab_size"], 4)
        e_ae_losses = torch.stack(e_ae_losses).mean()
        e_attn_losses = torch.stack(e_attn_losses).mean()
        # d_ae_losses = torch.stack(d_ae_losses).mean()
        # d_attn_losses = torch.stack(d_attn_losses).mean()
        #
        # # TODO log example of src and outs sometimes
        if self.step % config["train"]["after_mb_log_examples"] == 0:
            self.log_examples(srcs, trgs, outs, trg_ys, original_latents, z, reconstructed_latents)

        trg_ys = trg_ys.permute(1, 0, 3, 2)
        trg_ys = trg_ys.reshape(config["train"]["batch_size"], -1, 4)
        loss, loss_items = self.loss_computer(outs, trg_ys)
        # if self.step == 0:
        #     self.test_losses(loss, e_attn_losses, e_ae_losses, d_attn_losses, d_ae_losses, latents_reconstruction_loss)

        if self.encoder.training:
            self.model_optimizer.zero_grad()
            self.compressor_optimizer.zero_grad()
            #  (loss + e_attn_losses + e_ae_losses + d_attn_losses + d_ae_losses + latents_reconstruction_loss).backward()
            # (loss + e_attn_losses + e_ae_losses + latents_reconstruction_loss).backward()
            (loss + e_attn_losses + e_ae_losses).backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.compressor.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.decompressor.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
            self.compressor_optimizer.step()
            self.model_optimizer.optimize()  # TODO add decreasing optimization updating

        # losses = (loss.item(), e_attn_losses.item(), e_ae_losses.item(), d_attn_losses.item(), d_ae_losses.item(),
        #           *loss_items, latents_reconstruction_loss.item())
        losses = (loss.item(), e_attn_losses.item(), e_ae_losses.item(), 0, 0,
                  *loss_items, latents_reconstruction_loss.item())  # TODO adjust values
        if not config["train"]["aae"]:
            return losses
        was_training = self.encoder.training
        # TODO adversarial autoencoder
        self.model_optimizer.zero_grad()
        self.discriminator.zero_grad()
        eps = 1e-15
        # TODO Train discriminator
        self.encoder.eval()
        self.compressor.eval()
        self.decompressor.eval()
        self.decoder.eval()
        latents = []
        for seq in src:
            latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.encoder(seq, e_mems, e_cmems)
            latents.append(latent)
        latents = torch.stack(latents, dim=2)  # 1 x 4 x 10 x 301 x 32
        latents = self.compressor(latents)  # 1 x z_dim
        z_real_gauss = Variable(torch.randn_like(latents)).to(config["train"]["device"])  # TODO scale
        d_real_gauss = self.discriminator(z_real_gauss)
        d_fake_gauss = self.discriminator(latents)
        d_loss = -torch.mean(torch.log(d_real_gauss + eps) + torch.log(1 - d_fake_gauss + eps))
        if was_training:
            d_loss.backward()
            self.discriminator_optimizer.step()
        # TODO Train generator
        if was_training:
            self.encoder.train()
            self.compressor.train()
        latents = []
        for seq in src:
            latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.encoder(seq, e_mems, e_cmems)
            latents.append(latent)
        latents = torch.stack(latents, dim=2)  # 1 x 4 x 10 x 301 x 32
        latents = self.compressor(latents)  # 1 x z_dim
        d_fake_gauss = self.discriminator(latents)
        g_loss = -torch.mean(torch.log(d_fake_gauss + eps))
        if was_training:
            g_loss.backward()
            self.generator_optimizer.step()
        if was_training:
            self.decompressor.train()
            self.decoder.train()
        losses = losses + (d_loss.item(), g_loss.item())
        return losses

    def log_to_wandb(self, losses):
        mode = "train/" if self.encoder.training else "eval/"
        log = {"stuff/lr": self.model_optimizer.lr,
               mode + "loss": losses[0],
               mode + "encoder attention loss": losses[1],
               mode + "encoder autoencoder loss": losses[2],
               mode + "decoder attention loss": losses[3],
               mode + "decoder autoencoder loss": losses[4],
               mode + "drums loss": losses[5],
               mode + "bass loss": losses[6],
               mode + "guitar loss": losses[7],
               mode + "strings loss": losses[8],
               mode + "latents reconstruction loss": losses[9]}
        if config["train"]["aae"]:
            log[mode + "discriminator_loss"] = losses[10]
            log[mode + "generator_loss"] = losses[11]
        wandb.log(log)

    def reconstruct(self, batch, note_manager):
        srcs, _, src_masks, _, _ = batch
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 1)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"]).transpose(0, 1)
        e_mems, e_cmems, d_mems, d_cmems = self.get_memories()
        e_attn_losses = []
        e_ae_losses = []
        d_attn_losses = []
        d_ae_losses = []
        latents = []
        outs = []
        for src, src_mask in zip(srcs, src_masks):
            latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.encoder(src, src_mask, e_mems, e_cmems)
            e_attn_losses.append(e_attn_loss)
            e_ae_losses.append(e_ae_loss)
            latents.append(latent)
        latents = torch.stack(latents, dim=2)  # 1 x 4 x 10 x 301 x 32
        latents = self.compressor(latents)  # 1 x z_dim
        latents = self.decompressor(latents)
        latents = latents.transpose(0, 2)
        for latent, src_mask in zip(latents, src_masks):
            trg, trg_mask = self.greedy_decoding(latent, src_mask, d_mems, d_cmems)
            out, d_mems, d_cmems, d_attn_loss, d_ae_loss = self.decoder(trg, latent, src_mask, trg_mask, d_mems,
                                                                        d_cmems)
            d_attn_losses.append(d_attn_loss)
            d_ae_losses.append(d_ae_loss)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        outs = outs.reshape(config["train"]["batch_size"], -1, config["tokens"]["vocab_size"], 4)
        outs = outs.permute(0, 3, 1, 2)
        outs = torch.max(outs, dim=-1).indices
        outs = outs[0]
        srcs = srcs.permute(1, 2, 0, 3)
        srcs = srcs.reshape(config["train"]["batch_size"], 4, -1)
        srcs = srcs[0]
        original = note_manager.reconstruct_music(srcs.cpu().numpy())
        reconstructed = note_manager.reconstruct_music(outs.cpu().numpy())
        reconstructed = note_manager.cut_song(reconstructed, original.get_end_time())
        return original, reconstructed

    def generate(self, note_manager):
        latents = torch.randn((1, config["model"]["z_tot_dim"])).to(config["train"]["device"])
        latents = self.decompressor(latents)
        latents = latents.transpose(0, 2)
        _, _, d_mems, d_cmems = self.get_memories()
        outs = []
        for latent in latents:
            seed = torch.empty(4, config["train"]["batch_size"], 0, dtype=torch.long).to(config["train"]["device"])
            for _ in range(config["model"]["seq_len"]):
                out, _, _, _, _ = self.decoder(latent, seed, d_mems, d_cmems)
                out = torch.max(out, dim=-2).indices
                out = out.transpose(1, 2).transpose(0, 1)
                seed = torch.cat((seed, out[..., -1:]), dim=-1)
            out, d_mems, d_cmems, _, _ = self.decoder(latent, seed, d_mems, d_cmems)
            out = torch.max(out, dim=-2).indices
            out = out.transpose(1, 2)
            outs.append(out[..., 1:])
        outs = torch.stack(outs, dim=2)
        outs = outs.reshape(config["train"]["batch_size"], 4, -1)
        outs = outs[0]
        generated = note_manager.reconstruct_music(outs.cpu().numpy())
        return generated

    def train(self):
        # Create checkpoint folder
        timestamp = str(datetime.now())
        timestamp = timestamp[:timestamp.index('.')]
        timestamp = timestamp.replace(' ', '_').replace(':', '-')
        self.save_path = timestamp
        os.mkdir(self.save_path)
        # Models
        self.encoder = CompressiveEncoder().to(config["train"]["device"])
        self.compressor = LatentsCompressor().to(config["train"]["device"])
        self.decompressor = LatentsDecompressor().to(config["train"]["device"])
        self.decoder = CompressiveDecoder().to(config["train"]["device"])
        if config["train"]["aae"]:
            self.discriminator = Discriminator(config["model"]["z_i_dim"],
                                               config["model"]["z_tot_dim"]).to(config["train"]["device"])
        # Optimizers
        self.model_optimizer = CTOpt(torch.optim.Adam([{"params": self.encoder.parameters()},
                                                       {"params": self.compressor.parameters()},
                                                       {"params": self.decompressor.parameters()},
                                                       {"params": self.decoder.parameters()}], lr=0),
                                     config["train"]["warmup_steps"],
                                     (config["train"]["lr_min"], config["train"]["lr_max"]),
                                     config["train"]["decay_steps"], config["train"]["minimum_lr"])  # TODO add other
        if config["train"]["aae"]:
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-6)
            self.generator_optimizer = torch.optim.Adam([{"params": self.encoder.parameters()},
                                                         {"params": self.compressor.parameters()}], lr=1e-7)
        self.compressor_optimizer = torch.optim.Adam([{"params": self.compressor.parameters()},
                                                      {"params": self.decompressor.parameters()}])
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
                               max_len=config["model"]["total_seq_len"],
                               batch_size=config["train"]["batch_size"],
                               n_workers=config["train"]["n_workers"])
        tr_loader, ts_loader = dataset.get_loaders()
        # Wandb
        wandb.login()
        wandb.init(project="MusAE", config=self.settings, name="remote" if remote else "local" + ' ' + self.save_path)
        wandb.watch(self.encoder)
        wandb.watch(self.compressor)
        wandb.watch(self.decompressor)
        wandb.watch(self.decoder)
        # Train
        self.encoder.train()
        self.compressor.train()
        self.decompressor.train()
        self.decoder.train()
        desc = "Train epoch " + str(self.epoch) + ", mb " + str(0)
        train_progress = tqdm(total=config["train"]["mb_before_eval"], position=0, leave=True, desc=desc)
        trained = False
        it_counter = 0
        self.step = 0
        best_ts_loss = float("inf")
        for self.epoch in range(config["train"]["n_epochs"]):  # repeat for each epoch
            for it, batch in enumerate(tr_loader):  # repeat for each mini-batch
                if it % config["train"]["mb_before_eval"] == 0 and trained and config["train"]["do_eval"]:  # TODO RMVE
                    train_progress.close()
                    ts_losses = []
                    self.encoder.eval()
                    self.compressor.eval()
                    self.decompressor.eval()
                    self.decoder.eval()
                    desc = "Eval epoch " + str(self.epoch) + ", mb " + str(it)
                    test = None
                    generated = None  # to suppress warning
                    for test in tqdm(ts_loader, position=0, leave=True, desc=desc):  # remember test losses
                        ts_loss = self.run_mb(test)
                        ts_losses.append(ts_loss)
                    final = ()  # average losses
                    for i in range(len(ts_losses[0])):  # for each loss value
                        aux = []
                        for loss in ts_losses:  # for each computed loss
                            aux.append(loss[i])
                        avg = sum(aux) / len(aux)
                        final = final + (avg,)
                    if final[0] < best_ts_loss:
                        new_model = os.path.join(self.save_path, self.model_name + '_' + str(self.epoch) + '.pt')
                        print("Saving best model in " + new_model + ", DO NOT INTERRUPT")
                        for filename in glob.glob(os.path.join(self.save_path, self.model_name + '*')):
                            os.remove(filename)
                        best_ts_loss = final[0]
                        # torch.save(self.encoder, os.path.join(self.save_path, "encoder_" + str(self.epoch) + '.pt'))
                        # torch.save(self.compressor,
                        #            os.path.join(self.save_path, "compressor_" + str(self.epoch) + '.pt'))
                        # torch.save(self.decompressor,
                        #            os.path.join(self.save_path, "decompressor_" + str(self.epoch) + '.pt'))
                        # torch.save(self.decoder, os.path.join(self.save_path, "decoder_" + str(self.epoch) + '.pt'))
                        if config["train"]["aae"]:
                            torch.save(self.discriminator,
                                       os.path.join(self.save_path, "discriminator_" + str(self.epoch) + '.pt'))
                        print("Model saved")
                    self.log_to_wandb(final)
                    # TODO reconstruction and generation
                    note_manager = NoteRepresentationManager()
                    if config["train"]["aae"]:
                        generated = self.generate(note_manager)  # generation
                    original, reconstructed = self.reconstruct(batch, note_manager)  # TODO watch
                    prefix = "epoch_" + str(self.epoch) + "_mb_" + str(it)
                    original.write_midi(os.path.join(wandb.run.dir, prefix + "_original.mid"))
                    reconstructed.write_midi(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"))
                    if config["train"]["aae"]:
                        generated.write_midi(os.path.join(wandb.run.dir, prefix + "_generated.mid"))
                    midi_to_wav(os.path.join(wandb.run.dir, prefix + "_original.mid"),
                                os.path.join(wandb.run.dir, prefix + "_original.wav"))
                    midi_to_wav(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"),
                                os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"))
                    if config["train"]["aae"]:
                        midi_to_wav(os.path.join(wandb.run.dir, prefix + "_generated.mid"),
                                    os.path.join(wandb.run.dir, prefix + "_generated.wav"))
                    songs = [
                        wandb.Audio(os.path.join(wandb.run.dir, prefix + "_original.wav"),
                                    caption="original", sample_rate=32),
                        wandb.Audio(os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"),
                                    caption="reconstructed", sample_rate=32)]
                    if config["train"]["aae"]:
                        songs.append(wandb.Audio(os.path.join(wandb.run.dir, prefix + "_generated.wav"),
                                                 caption="generated", sample_rate=32))
                    wandb.log({str(it_counter): songs})
                    it_counter += 1
                    self.encoder.train()
                    self.compressor.train()
                    self.decompressor.train()
                    self.decoder.train()
                    desc = "Train epoch " + str(self.epoch) + ", mb " + str(it)
                    train_progress = tqdm(total=config["train"]["mb_before_eval"], position=0, leave=True, desc=desc)
                tr_losses = self.run_mb(batch)
                self.log_to_wandb(tr_losses)
                train_progress.update()
                trained = True
                self.step += 1
