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



    @staticmethod
    def log_attn_images(mem_weights, self_weights, dec_src_weights):
        mem_img = mem_weights.detach().cpu().numpy()
        mem_formatted = (mem_img * 255 / np.max(mem_img)).astype('uint8')
        mem_img = Image.fromarray(mem_formatted)

        self_img = self_weights.detach().cpu().numpy()
        self_formatted = (self_img * 255 / np.max(self_img)).astype('uint8')
        self_img = Image.fromarray(self_formatted)

        src_img = dec_src_weights.detach().cpu().numpy()
        src_formatted = (src_img * 255 / np.max(src_img)).astype('uint8')
        src_img = Image.fromarray(src_formatted)

        pad = np.array([255 * (i % 2) for i in range(config["model"]["seq_len"] * 9)]).reshape(-1, 9).astype(np.uint8)
        pad_img = Image.fromarray(pad)
        # CONCATENATE IMAGES
        images = [mem_img, pad_img, self_img, pad_img, src_img]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        wandb.log({"examples": [wandb.Image(new_im, caption="Attention")]})

    @staticmethod
    def log_examples(e_in, d_in, pred, exp, lat=None, z=None, r_lat=None):
        enc_input = e_in.transpose(0, 2)[0].reshape(4, -1).detach().cpu().numpy()
        dec_input = d_in.transpose(0, 2)[0].reshape(4, -1).detach().cpu().numpy()
        predicted = torch.max(pred, dim=-2).indices.permute(0, 2, 1)[0].reshape(4, -1).detach().cpu().numpy()
        expected = exp[0].transpose(0, 1).detach().cpu().numpy()
        columns = ["Encoder Input: " + str(enc_input.shape),
                   "Decoder Input: " + str(dec_input.shape),
                   "Predicted: " + str(predicted.shape),
                   "Expected: " + str(expected.shape)]
        inputs = (enc_input[:, :10], dec_input[:, :10], predicted[:, :10], expected[:, :10])
        if lat is not None and type(lat) is not list:
            latent = lat.detach().cpu().numpy()
            inputs = inputs + (latent,)
            columns.append("Latents: " + str(latent.shape))
        if z is not None:
            zeta = z.detach().cpu().numpy()
            inputs = inputs + (zeta,)
            columns.append("Z: " + str(z.shape))
        if r_lat is not None:
            recon_l = r_lat.detach().cpu().numpy()
            inputs = inputs + (recon_l,)
            columns.append("Real latents: " + str(recon_l.shape))
        table = wandb.Table(columns=columns)
        table.add_data(*inputs)
        wandb.log({"out": table})

    @staticmethod
    def log_memories(e_mems, e_cmems, d_mems, d_cmems):
        e_mems = e_mems.detach().cpu().numpy()
        e_cmems = e_cmems.detach().cpu().numpy()
        d_mems = d_mems.detach().cpu().numpy()
        d_cmems = d_cmems.detach().cpu().numpy()
        columns = ["Encoder Memory: " + str(e_mems.shape),
                   "Encoder Compressed memory: " + str(e_cmems.shape),
                   "Decoder Memory: " + str(d_mems.shape),
                   "Decoder Compressed memory: " + str(d_cmems.shape)]

        inputs = (e_mems[:, :, 0, :10, :], e_cmems[:, :, 0, :10, :], d_mems[:, :, 0, :10, :], d_cmems[:, :, 0, :10, :])
        table = wandb.Table(columns=columns)
        table.add_data(*inputs)
        wandb.log({"memories": table})

    @staticmethod
    def get_memories():
        a = 4
        b = config["model"]["layers"]
        c = config["train"]["batch_size"]
        e = config["model"]["d_model"]
        device = config["train"]["device"]
        mem_len = config["model"]["mem_len"]
        cmem_len = config["model"]["cmem_len"]
        e_mems = torch.zeros(a, b, c, mem_len, e, dtype=torch.float32, device=device)
        e_cmems = torch.zeros(a, b, c, cmem_len, e, dtype=torch.float32, device=device)
        d_mems = torch.zeros(a, b, c, mem_len, e, dtype=torch.float32, device=device)
        d_cmems = torch.zeros(a, b, c, cmem_len, e, dtype=torch.float32, device=device)
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

    @staticmethod
    def pad_attention(attentions):
        length = max([s.shape[-1] for s in attentions])
        for count, attention in enumerate(attentions):
            attentions[count] = f.pad(attention, (length - attention.shape[-1], 0))
        return torch.mean(torch.stack(attentions, dim=0), dim=0)

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
        e_mems, e_cmems, d_mems, d_cmems = self.get_memories()

        # Encode
        for src, src_mask in zip(srcs, src_masks):
            latent, e_mems, e_cmems, e_attn_loss, sw = self.encoder(src, src_mask, e_mems, e_cmems)
            enc_self_weights.append(sw)
            e_attn_losses.append(e_attn_loss)

        # Give only last memories to decoder
        # lat_mem = e_mems[:, -1, :, :, :]
        # # lat_mem = lat_mem.unsqueeze(1).repeat(1, config["model"]["layers"], 1, 1, 1)
        # lat_cmem = e_cmems[:, -1, :, :, :]
        # # lat_cmem = lat_cmem.unsqueeze(1).repeat(1, config["model"]["layers"], 1, 1, 1)
        # latent = torch.cat((lat_cmem, lat_mem), dim=-2)
        latent = self.latent_compressor(latent)
        latent = latent.transpose(0, 1)

        # Decode
        for trg, src_mask, trg_mask in zip(trgs, src_masks, trg_masks):
            out, self_weight, src_weight, d_mems, d_cmems, d_attn_loss = self.decoder(trg, trg_mask, src_mask, latent,
                                                                                      d_mems, d_cmems)
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
        if self.step % config["train"]["after_mb_log_attn_img"] == 0:
            enc_self_weights = self.pad_attention(enc_self_weights)
            dec_self_weights = self.pad_attention(dec_self_weights)
            dec_src_weights = self.pad_attention(dec_src_weights)
            self.log_attn_images(enc_self_weights, dec_self_weights, dec_src_weights)
        if self.step % config["train"]["after_mb_log_memories"] == 0:
            self.log_memories(e_mems, e_cmems, d_mems, d_cmems)
        if self.step % config["train"]["after_mb_log_examples"] == 0:
            self.log_examples(srcs, trgs, outs, trg_ys)
        if self.step == 0 and config["train"]["test_loss"]:
            self.test_losses(loss, e_attn_losses, d_attn_losses)

        # OPTIMIZE
        if self.encoder.training:
            self.model_optimizer.zero_grad()
            optimizing_losses = loss + e_attn_losses + d_attn_losses
            optimizing_losses.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.1)
            self.model_optimizer.optimize()  # TODO add decreasing optimization updating

        losses = (loss.item(), e_attn_losses.item(), d_attn_losses.item(), *loss_items)

        return losses

    def log_to_wandb(self, losses):
        mode = "train/" if self.encoder.training else "eval/"
        log = {"stuff/lr": self.model_optimizer.lr,
               mode + "loss": losses[0],
               mode + "encoder attention loss": losses[1],
               mode + "decoder attention loss": losses[2],
               mode + "drums loss": losses[3],
               mode + "guitar loss": losses[4],
               mode + "bass loss": losses[5],
               mode + "strings loss": losses[6]}
        if config["train"]["aae"]:
            log[mode + "discriminator_loss"] = losses[7]
            log[mode + "generator_loss"] = losses[8]
        wandb.log(log)

    def reconstruct(self, batch, note_manager):
        srcs, trgs, src_masks, trg_masks, _ = batch
        srcs = torch.LongTensor(srcs.long()).to(config["train"]["device"]).transpose(0, 2)
        trgs = torch.LongTensor(trgs.long()).to(config["train"]["device"]).transpose(0, 2)
        src_masks = torch.BoolTensor(src_masks).to(config["train"]["device"]).transpose(0, 2)
        trg_masks = torch.BoolTensor(trg_masks).to(config["train"]["device"]).transpose(0, 2)
        e_mems, e_cmems, d_mems, d_cmems = self.get_memories()
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
        # What are we doing?
        cmem_range = config["model"]["cmem_len"]*config["model"]["cmem_ratio"]
        max_range = config["model"]["layers"]*(cmem_range+config["model"]["mem_len"])
        given = config["train"]["truncated_bars"]*config["model"]["seq_len"]
        print("Giving ", given, " as input to model with a maximum range of ", max_range)
        # Create checkpoint folder
        timestamp = str(datetime.now())
        timestamp = timestamp[:timestamp.index('.')]
        timestamp = timestamp.replace(' ', '_').replace(':', '-')
        self.save_path = timestamp
        os.mkdir(self.save_path)
        # Models
        self.encoder = CompressiveEncoder().to(config["train"]["device"])
        self.latent_compressor = LatentCompressor(config["model"]["d_model"]).to(config["train"]["device"])
        self.decoder = CompressiveDecoder().to(config["train"]["device"])
        # if config["train"]["aae"]:
        #     self.discriminator = Discriminator(config["model"]["z_i_dim"],
        #                                        config["model"]["z_tot_dim"]).to(config["train"]["device"])
        # Optimizers
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
        print("Giving ", len(tr_loader), " training samples and ", len(ts_loader), " test samples")
        # Wandb
        wandb.login()
        wandb.init(project="MusAE", config=self.settings, name="remote" if remote else "local" + ' ' + self.save_path)
        wandb.watch(self.encoder)
        wandb.watch(self.latent_compressor)
        wandb.watch(self.decoder)
        # Train
        self.encoder.train()
        self.latent_compressor.train()
        self.decoder.train()
        desc = "Train epoch " + str(self.epoch) + ", mb " + str(0)
        train_progress = tqdm(total=config["train"]["mb_before_eval"], position=0, leave=True, desc=desc)
        it_counter = 0
        self.step = 0
        best_ts_loss = float("inf")
        for self.epoch in range(config["train"]["n_epochs"]):  # for each epoch
            for song_it, batch in enumerate(tr_loader):  # for each song
                for bar_it in range(batch[0].shape[2]):  # for each bar groups
                    # Train
                    mb = ()
                    for elem in batch:  # create mb
                        mb = mb + (elem[:, :, bar_it, ...],)
                    tr_losses = self.run_mb(mb)
                    self.log_to_wandb(tr_losses)
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
                        for test in tqdm(ts_loader, position=0, leave=True, desc=desc):  # remember test losses
                            for i in range(test[0].shape[2]):
                                mb = ()
                                for elem in test:  # create mb
                                    mb = mb + (elem[:, :, i, ...],)
                            ts_loss = self.run_mb(mb)
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
                            torch.save(self.encoder, os.path.join(self.save_path, "encoder_" + str(self.epoch) + '.pt'))
                            # torch.save(self.latent_compressor, os.path.join(self.save_path, TODO save compressor
                            #                                                 "latent_compressor"+str(self.epoch)+'.pt'))
                            torch.save(self.decoder, os.path.join(self.save_path, "decoder_" + str(self.epoch) + '.pt'))
                            # if config["train"]["aae"]:
                            #     torch.save(self.discriminator,
                            #                os.path.join(self.save_path, "discriminator_" + str(self.epoch) + '.pt'))
                            print("Model saved")
                        self.log_to_wandb(final)
                        # TODO reconstruction and generation
                        note_manager = NoteRepresentationManager()
                        # if config["train"]["aae"]:
                        #     generated = self.generate(note_manager)  # generation
                        original, reconstructed = self.reconstruct(mb, note_manager)  # TODO just one bar?
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

                # for i in range(batch[0].shape[2]):  # repeat for each bar groups
                #
                #     # n_tokens = 0 TODO useful only with few bars
                #     # n_tokens += torch.numel(batch[0][:, :, i, ...]) - \
                #     #             (batch[0][:, :, i, ...] == config["tokens"]["pad"]).sum().item() - \
                #     #             (batch[0][:, :, i, ...] == config["tokens"]["sos"]).sum().item() - \
                #     #             (batch[0][:, :, i, ...] == config["tokens"]["eos"]).sum().item()
                #     # if n_tokens == 0:
                #     #     print("Empty bars skipped")
                #     #     continue
                #     mb = ()
                #     for elem in batch:  # create mb
                #         mb = mb + (elem[:, :, i, ...],)
                #     tr_losses = self.run_mb(mb)
                #     self.log_to_wandb(tr_losses)
                #     train_progress.update()
                #     trained = True
                #     self.step += 1

        # if not config["train"]["aae"]:
        #     return losses
        # was_training = self.encoder.training
        # # TODO adversarial autoencoder
        # self.model_optimizer.zero_grad()
        # self.discriminator.zero_grad()
        # eps = 1e-15
        # # TODO Train discriminator
        # self.encoder.eval()
        # self.compressor.eval()
        # self.decompressor.eval()
        # self.decoder.eval()
        # latents = []
        # for seq in src:
        #     latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.encoder(seq, e_mems, e_cmems)
        #     latents.append(latent)
        # latents = torch.stack(latents, dim=2)  # 1 x 4 x 10 x 301 x 32
        # latents = self.compressor(latents)  # 1 x z_dim
        # z_real_gauss = Variable(torch.randn_like(latents)).to(config["train"]["device"])  # TODO scale
        # d_real_gauss = self.discriminator(z_real_gauss)
        # d_fake_gauss = self.discriminator(latents)
        # d_loss = -torch.mean(torch.log(d_real_gauss + eps) + torch.log(1 - d_fake_gauss + eps))
        # if was_training:
        #     d_loss.backward()
        #     self.discriminator_optimizer.step()
        # # TODO Train generator
        # if was_training:
        #     self.encoder.train()
        #     self.compressor.train()
        # latents = []
        # for seq in src:
        #     latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.encoder(seq, e_mems, e_cmems)
        #     latents.append(latent)
        # latents = torch.stack(latents, dim=2)  # 1 x 4 x 10 x 301 x 32
        # latents = self.compressor(latents)  # 1 x z_dim
        # d_fake_gauss = self.discriminator(latents)
        # g_loss = -torch.mean(torch.log(d_fake_gauss + eps))
        # if was_training:
        #     g_loss.backward()
        #     self.generator_optimizer.step()
        # if was_training:
        #     self.decompressor.train()
        #     self.decoder.train()
        # losses = losses + (d_loss.item(), g_loss.item())
        # return losses

    # TODO latents compression with loss
    # latents = torch.stack(latents, dim=2)  # 1 x 4 x 10 x 301 x 32
    # original_latents = None
    # z = None
    # reconstruction_loss = 0
    # # if self.compress:
    # #     import copy
    # #     loss_function = torch.nn.MSELoss()
    # # original_latents = copy.deepcopy(latents.data)
    # z = self.compressor(latents)  # 1 x z_dim
    # latents = self.decompressor(z)
    # # reconstruction_loss = loss_function(original_latents, latents)
    # # original_latents = original_latents.transpose(0, 2)
    # # if not self.final_stage:
    # #     self.compressor_optimizer.zero_grad()
    # #     reconstruction_loss.backward()
    # #     torch.nn.utils.clip_grad_norm_(self.compressor.parameters(), 0.1)
    # #     torch.nn.utils.clip_grad_norm_(self.decompressor.parameters(), 0.1)
    # #     self.compressor_optimizer.step()
    # # reconstruction_loss = reconstruction_loss.item()
