import os
import torch
from datetime import datetime
from tqdm.auto import tqdm
from config import config, remote
from iterate_dataset import SongIterator
from optimizer import CTOpt
from label_smoother import LabelSmoothing
from loss_computer import SimpleLossCompute
from create_bar_dataset import NoteRepresentationManager
import glob
import wandb
from midi_converter import midi_to_wav
from discriminator import Discriminator
from compressive_transformer import CompressiveEncoder, CompressiveDecoder
from compress_latents import LatentCompressor
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from logger import Logger
from utilities import get_memories, create_trg_mask, pad_attention


class Trainer:
    def __init__(self):
        self.logger = None
        self.save_path = None
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
        self.compress = False
        self.final_stage = False
        self.latent_compressor = None

    def test_losses(self, loss, e_attn_losses, d_attn_losses):
        losses = [loss, e_attn_losses, d_attn_losses]
        names = ["loss", "e_att_losses", "d_attn_losses"]
        for ls, name in zip(losses, names):
            print("********************** Optimized by " + name)
            self.model_optimizer.optimizer.zero_grad(set_to_none=True)
            ls.backward(retain_graph=True)
            for model in [self.encoder, self.decoder]:  # removed latent compressor
                for module_name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        print(module_name)
        self.model_optimizer.optimizer.zero_grad(set_to_none=True)
        (losses[0] + losses[1] + losses[2]).backward(retain_graph=True)
        print("********************** NOT OPTIMIZED BY NOTHING")
        for model in [self.encoder, self.decoder]:  # removed latent compressor
            for module_name, parameter in model.named_parameters():
                if parameter.grad is None:
                    print(module_name)

    def generate(self, note_manager):
        latents = torch.randn((1, config["model"]["z_tot_dim"])).to(config["train"]["device"])
        latents = self.decompressor(latents)
        latents = latents.transpose(0, 2)
        _, _, d_mems, d_cmems = get_memories()
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

    def greedy_decoding(self, latent, src_mask, d_mems, d_cmems):
        trg = np.full((config["train"]["batch_size"], 4, 1), config["tokens"]["sos"])
        for _ in range(config["model"]["seq_len"] - 1):
            trg_mask = create_trg_mask(trg)
            trg = torch.LongTensor(trg).to(config["train"]["device"])
            out, _, _, _, _ = self.decoder(trg, latent, src_mask, trg_mask, d_mems, d_cmems)
            out = torch.max(out, dim=-2).indices
            out = out.transpose(1, 2)
            trg = torch.cat((trg, out[..., -1:]), dim=-1)
            trg = trg.detach().cpu().numpy()
        trg_mask = create_trg_mask(trg)
        trg = torch.LongTensor(trg).to(config["train"]["device"])
        return trg, trg_mask

    def reconstruct(self, batch, note_manager):
        srcs, trgs, src_masks, trg_masks, _ = batch
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 2)
        trgs = torch.LongTensor(trgs.long()).to(config["train"]["device"]).transpose(0, 2)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"]).transpose(0, 2)
        trg_masks = torch.BoolTensor(trg_masks).to(config["train"]["device"]).transpose(0, 2)
        e_mems, e_cmems, d_mems, d_cmems = get_memories()
        outs = []
        latent = None
        for src, src_mask in zip(srcs, src_masks):
            latent, e_mems, e_cmems, _, _ = self.encoder(src, src_mask, e_mems, e_cmems)
        latent = self.latent_compressor(latent)  # 1 x z_dim
        latent = latent.transpose(0, 1)
        for trg, src_mask, trg_mask in zip(trgs, src_masks, trg_masks):
            # trg, trg_mask = self.greedy_decoding(latent, src_mask, d_mems, d_cmems)
            out, _, _, d_mems, d_cmems, _ = self.decoder(trg, trg_mask, src_mask, latent, d_mems, d_cmems)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        # outs = outs.reshape(config["train"]["batch_size"], -1, config["tokens"]["vocab_size"], 4)
        outs = outs.permute(0, 4, 1, 2, 3)
        outs = torch.max(outs, dim=-1).indices
        outs = outs[0]
        srcs = srcs.permute(2, 1, 0, 3)
        # srcs = srcs.reshape(config["train"]["batch_size"], 4, -1)
        srcs = srcs[0]
        original = note_manager.reconstruct_music(srcs.cpu().numpy())
        reconstructed = note_manager.reconstruct_music(outs.cpu().numpy())
        # reconstructed = note_manager.cut_song(reconstructed, original.get_end_time())
        return original, reconstructed

    def run_mb(self, batch):
        # SETUP VARIABLES
        srcs, trgs, src_masks, trg_masks, trg_ys = batch
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 2)
        trgs = torch.LongTensor(trgs.long()).to(config["train"]["device"]).transpose(0, 2)  # invert batch and seq
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
            e_mems = e_mems.detach()
            e_cmems = e_cmems.detach()
            enc_self_weights.append(sw)
            e_attn_losses.append(e_attn_loss)

        latent = self.latent_compressor(latent)
        latent = latent.transpose(0, 1)

        # Decode
        for trg, src_mask, trg_mask in zip(trgs, src_masks, trg_masks):
            out, self_weight, src_weight, d_mems, d_cmems, d_attn_loss = self.decoder(trg, trg_mask, src_mask, latent,
                                                                                      d_mems, d_cmems)
            d_mems = d_mems.detach()
            d_cmems = d_cmems.detach()
            d_attn_losses.append(d_attn_loss)
            outs.append(out)
            dec_self_weights.append(self_weight)
            dec_src_weights.append(src_weight)

        outs = torch.stack(outs, dim=1)
        outs = outs.reshape(config["train"]["batch_size"], -1, config["tokens"]["vocab_size"], 4)
        e_attn_losses = torch.stack(e_attn_losses).mean()
        d_attn_losses = torch.stack(d_attn_losses).mean()

        trg_ys = trg_ys.permute(2, 0, 3, 1)
        trg_ys = trg_ys.reshape(config["train"]["batch_size"], -1, 4)
        loss, loss_items = self.loss_computer(outs, trg_ys)

        # SOME TESTS
        if self.encoder.training:
            if self.step % config["train"]["after_mb_log_attn_img"] == 0:
                # enc_self_weights = pad_attention(enc_self_weights)
                # dec_self_weights = pad_attention(dec_self_weights)
                # dec_src_weights = pad_attention(dec_src_weights)
                enc_self_weights = torch.stack(enc_self_weights)
                dec_self_weights = torch.stack(dec_self_weights)
                dec_src_weights = torch.stack(dec_src_weights)
                self.logger.log_attn_heatmap(enc_self_weights, dec_self_weights, dec_src_weights)
            if self.step % config["train"]["after_mb_log_memories"] == 0:
                self.logger.log_memories(e_mems, e_cmems, d_mems, d_cmems)
            if self.step % config["train"]["after_mb_log_examples"] == 0:
                self.logger.log_examples(srcs, trgs, outs, trg_ys)
                self.logger.log_latent(latent)
            if self.step == 0 and config["train"]["test_loss"]:
                self.test_losses(loss, e_attn_losses, d_attn_losses)

        # OPTIMIZE
        if self.encoder.training:
            self.model_optimizer.zero_grad()
            optimizing_losses = loss + e_attn_losses + d_attn_losses
            optimizing_losses.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
            self.model_optimizer.optimize()

        losses = (loss.item(), e_attn_losses.item(), d_attn_losses.item(), *loss_items)

        return losses

    def train(self):

        # Create checkpoint folder
        timestamp = str(datetime.now())
        timestamp = timestamp[:timestamp.index('.')]
        timestamp = timestamp.replace(' ', '_').replace(':', '-')
        self.save_path = timestamp
        os.mkdir(self.save_path)

        # Create models
        self.encoder = CompressiveEncoder().to(config["train"]["device"])
        self.latent_compressor = LatentCompressor(config["model"]["d_model"]).to(config["train"]["device"])
        self.decoder = CompressiveDecoder().to(config["train"]["device"])
        # if config["train"]["aae"]:
        #     self.discriminator = Discriminator(config["model"]["z_i_dim"],
        #                                        config["model"]["z_tot_dim"]).to(config["train"]["device"])

        # Create optimizers
        self.model_optimizer = CTOpt(torch.optim.Adam([{"params": self.encoder.parameters()},
                                                       {"params": self.latent_compressor.parameters()},
                                                       {"params": self.decoder.parameters()}], lr=0),
                                     config["train"]["warmup_steps"],
                                     (config["train"]["lr_min"], config["train"]["lr_max"]),
                                     config["train"]["decay_steps"], config["train"]["minimum_lr"])  # TODO add other
        # if config["train"]["aae"]:
        #     self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-6)
        #     self.generator_optimizer = torch.optim.Adam([{"params": self.encoder.parameters()},
        #                                                  {"params": self.compressor.parameters()}], lr=1e-7)
        # self.compressor_optimizer = torch.optim.Adam([{"params": self.compressor.parameters()},
        #                                               {"params": self.decompressor.parameters()}])

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
        wandb.init(project="MusAE", config=config, name="r_"+self.save_path if remote else "l_"+self.save_path)
        wandb.watch(self.encoder)
        wandb.watch(self.latent_compressor)
        wandb.watch(self.decoder)

        # Print info about training
        cmem_range = config["model"]["cmem_len"]*config["model"]["cmem_ratio"]
        max_range = config["model"]["layers"]*(cmem_range+config["model"]["mem_len"])
        given = config["train"]["truncated_bars"]*config["model"]["seq_len"]
        assert given <= max_range, "Given {} as input to model with max range {}".format(given, max_range)
        print("Giving ", given, " as input to model with a maximum range of ", max_range)
        print("Giving ", len(tr_loader), " training samples and ", len(ts_loader), " test samples")
        print("Giving ", config["train"]["truncated_bars"], " bars to a model with ",
              config["model"]["layers"], " layers")

        # Train
        self.encoder.train()
        self.latent_compressor.train()
        self.decoder.train()
        desc = "Train epoch " + str(self.epoch) + ", mb " + str(0)
        train_progress = tqdm(total=config["train"]["mb_before_eval"], position=0, leave=True, desc=desc)
        it_counter = 0
        self.step = 0  # TODO -1 to make eval as first thing
        best_ts_loss = float("inf")
        for self.epoch in range(config["train"]["n_epochs"]):  # for each epoch
            for song_it, batch in enumerate(tr_loader):  # for each song
                for bar_it in range(batch[0].shape[2]):  # for each bar groups

                    # Train
                    n_tokens = 0  # useful only with few bars
                    n_tokens += torch.numel(batch[0][:, :, bar_it, ...]) - \
                                (batch[0][:, :, bar_it, ...] == config["tokens"]["pad"]).sum().item() - \
                                (batch[0][:, :, bar_it, ...] == config["tokens"]["sos"]).sum().item() - \
                                (batch[0][:, :, bar_it, ...] == config["tokens"]["eos"]).sum().item()
                    if n_tokens == 0:
                        print("Empty bars skipped")
                        continue
                    mb = ()
                    for elem in batch:  # create mb
                        mb = mb + (elem[:, :, bar_it, ...],)
                    tr_losses = self.run_mb(mb)
                    self.logger.log_losses(tr_losses, self.model_optimizer.lr, self.encoder.training)
                    train_progress.update()
                    self.step += 1

                    # Eval
                    if self.step % config["train"]["mb_before_eval"] == 0 and config["train"]["do_eval"]:
                        train_progress.close()
                        ts_losses = []
                        self.encoder.eval()
                        self.latent_compressor.eval()
                        self.decoder.eval()
                        desc = "Eval epoch " + str(self.epoch) + ", mb " + str(song_it)
                        # test = None
                        # generated = None  # to suppress warning

                        # Compute validation score
                        for test in tqdm(ts_loader, position=0, leave=True, desc=desc):  # remember test losses
                            for i in range(test[0].shape[2]):
                                n_tokens = 0  # useful only with few bars
                                n_tokens += torch.numel(test[0][:, :, i, ...]) - \
                                            (test[0][:, :, i, ...] == config["tokens"]["pad"]).sum().item() - \
                                            (test[0][:, :, i, ...] == config["tokens"]["sos"]).sum().item() - \
                                            (test[0][:, :, i, ...] == config["tokens"]["eos"]).sum().item()
                                if n_tokens == 0:
                                    print("Empty bars skipped")
                                    continue
                                ts_mb = ()
                                for elem in test:  # create mb
                                    ts_mb = ts_mb + (elem[:, :, i, ...],)
                                ts_loss = self.run_mb(ts_mb)
                                ts_losses.append(ts_loss)
                        final = ()  # average losses
                        for i in range(len(ts_losses[0])):  # for each loss value
                            aux = []
                            for loss in ts_losses:  # for each computed loss
                                aux.append(loss[i])
                            avg = sum(aux) / len(aux)
                            final = final + (avg,)

                        # Save best models
                        if final[0] < best_ts_loss:
                            print("Saving best model in " + self.save_path + ", DO NOT INTERRUPT")
                            for filename in glob.glob(os.path.join(self.save_path, '*')):
                                os.remove(filename)
                            best_ts_loss = final[0]
                            torch.save(self.encoder, os.path.join(self.save_path, "encoder_" + str(self.step) + '.pt'))
                            torch.save(self.latent_compressor, os.path.join(self.save_path,
                                                                            "latent_compressor_"+str(self.step)+'.pt'))
                            torch.save(self.decoder, os.path.join(self.save_path, "decoder_" + str(self.step) + '.pt'))
                            # if config["train"]["aae"]:
                            #     torch.save(self.discriminator,
                            #                os.path.join(self.save_path, "discriminator_" + str(self.epoch) + '.pt'))
                            print("Model saved")
                        self.logger.log_losses(final, self.model_optimizer.lr, self.encoder.training)

                        # Reconstruction and generation
                        note_manager = NoteRepresentationManager()
                        # if config["train"]["aae"]:
                        #     generated = self.generate(note_manager)  # generation
                        to_reconstruct = ts_mb if remote else mb
                        original, reconstructed = self.reconstruct(to_reconstruct, note_manager)  # TODO reconstruct
                        prefix = "epoch_" + str(self.epoch) + "_mb_" + str(song_it)
                        original.write_midi(os.path.join(wandb.run.dir, prefix + "_original.mid"))
                        reconstructed.write_midi(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"))
                        # if config["train"]["aae"]:
                        #     generated.write_midi(os.path.join(wandb.run.dir, prefix + "_generated.mid"))
                        midi_to_wav(os.path.join(wandb.run.dir, prefix + "_original.mid"),
                                    os.path.join(wandb.run.dir, prefix + "_original.wav"))
                        midi_to_wav(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"),
                                    os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"))
                        # if config["train"]["aae"]:
                        #     midi_to_wav(os.path.join(wandb.run.dir, prefix + "_generated.mid"),
                        #                 os.path.join(wandb.run.dir, prefix + "_generated.wav"))
                        songs = [
                            wandb.Audio(os.path.join(wandb.run.dir, prefix + "_original.wav"),
                                        caption="original", sample_rate=32),
                            wandb.Audio(os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"),
                                        caption="reconstructed", sample_rate=32)]
                        # if config["train"]["aae"]:
                        #     songs.append(wandb.Audio(os.path.join(wandb.run.dir, prefix + "_generated.wav"),
                        #                              caption="generated", sample_rate=32))
                        wandb.log({"validation reconstruction example": songs})
                        it_counter += 1
                        self.encoder.train()
                        self.latent_compressor.train()
                        self.decoder.train()
                        desc = "Train epoch " + str(self.epoch) + ", mb " + str(song_it)
                        train_progress = tqdm(total=config["train"]["mb_before_eval"], position=0, leave=True, desc=desc)
