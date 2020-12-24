import os
import torch
from datetime import datetime
from tqdm.auto import tqdm
from config import config, remote
from iterate_dataset import SongIterator
from optimizer import CTOpt
from label_smoother import LabelSmoothing
from loss_computer import SimpleLossCompute
import numpy as np
from create_dataset import NoteRepresentationManager
import glob
import wandb
from midi_converter import midi_to_wav
from discriminator import Discriminator
from compress_latents import CompressLatents, DecompressLatents


class Trainer:
    def __init__(self, model=None, save_path=None, model_name="checkpoint", plot_name="plot", settings=config):
        self.save_path = save_path
        self.model_name = model_name
        self.plot_name = plot_name
        self.settings = settings
        self.epoch = 0
        self.loss_computer = None
        self.optimizer = None
        # Models
        self.model = model
        self.compress_latents = CompressLatents(in_dim=config["model"]["d_model"],
                                                z_dim=config["model"]["z_dim"],
                                                n_bars=config["data"]["max_bars"]).to(config["train"]["device"])
        self.decompress_latents = DecompressLatents(in_dim=config["model"]["d_model"],
                                                    z_dim=config["model"]["z_dim"],
                                                    n_bars=config["data"]["max_bars"]).to(config["train"]["device"])
        self.discriminator = Discriminator(config["data"]["max_bars"] * config["model"]["d_model"],
                                           config["model"]["z_dim"]).to(config["train"]["device"])
        # Optimizers
        self.model_optimizer = CTOpt(torch.optim.Adam([{"params": self.model.parameters()},
                                                       {"params": self.compress_latents.parameters()},
                                                       {"params": self.decompress_latents.parameters()}], lr=0),
                                     config["train"]["warmup_steps"],
                                     (config["train"]["lr_min"], config["train"]["lr_max"]))  # TODO add other
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=5e-5)
        # Adversarial part
        self.discriminator = None
        self.discriminator_optimizer = None

    @staticmethod
    def get_memories():
        """
        :return: 4 empty memories with dimension taken from attributes of the self class
        """
        a = config["model"]["n_tracks"]
        b = config["model"]["layers"]
        c = config["train"]["batch_size"]
        e = config["model"]["d_model"]
        # mem_len = config["model"]["mem_len"]
        # cmem_len = config["model"]["cmem_len"]
        device = config["train"]["device"]
        e_mems = torch.zeros(a, b, c, 0, e, dtype=torch.float32, device=device)
        e_cmems = torch.zeros(a, b, c, 0, e, dtype=torch.float32, device=device)
        d_mems = torch.zeros(a, b, c, 0, e, dtype=torch.float32, device=device)
        d_cmems = torch.zeros(a, b, c, 0, e, dtype=torch.float32, device=device)
        return e_mems, e_cmems, d_mems, d_cmems

    def run_mb(self, src):
        """
        It runs a mini-batch and return loss, if the model is in training it does also optimization
        :param src: Tensor with shape (n_batch, time_steps)
        :return: Tuple with total_loss, auxiliary_losses and instruments losses
        """
        # Train reconstruction
        src = np.swapaxes(src, 0, 2)  # swap batch, tracks -> tracks, batch
        bars = torch.LongTensor(src.long()).to(config["train"]["device"])
        e_mems, e_cmems, d_mems, d_cmems = self.get_memories()
        e_attn_losses = []
        e_ae_losses = []
        d_attn_losses = []
        d_ae_losses = []
        latents = []
        outs = []
        for bar in bars:
            latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.model.encode(bar, e_mems, e_cmems)
            e_attn_losses.append(e_attn_loss)
            e_ae_losses.append(e_ae_loss)
            latents.append(latent)
        latents = torch.stack(latents)
        latents = self.compress_latents(latents)
        latents = self.decompress_latents(latents)
        for bar, latent in zip(bars, latents):
            out, d_mems, d_cmems, d_attn_loss, d_ae_loss = self.model.decode(latent, bar, d_mems, d_cmems)
            d_attn_losses.append(d_attn_loss)
            d_ae_losses.append(d_ae_loss)
            outs.append(out)
        outs = torch.stack(outs)
        e_ae_losses = torch.stack(e_ae_losses).mean()
        e_attn_losses = torch.stack(e_attn_losses).mean()
        d_ae_losses = torch.stack(d_ae_losses).mean()
        d_attn_losses = torch.stack(d_attn_losses).mean()
        n_tokens = torch.count_nonzero(bars)
        n_tokens_drums = torch.count_nonzero(bars[:, 0, :, :])
        n_tokens_bass = torch.count_nonzero(bars[:, 1, :, :])
        n_tokens_guitar = torch.count_nonzero(bars[:, 2, :, :])
        n_tokens_strings = torch.count_nonzero(bars[:, 3, :, :])
        norm = (n_tokens, n_tokens_drums, n_tokens_bass, n_tokens_guitar, n_tokens_strings)
        bars = bars.transpose(1, 2)
        bars = bars.transpose(2, 3)
        bars = bars[:, :, 1:, :]
        loss, loss_items = self.loss_computer(outs, bars, norm)  # TODO pad token and model output...?
        if self.model.training:
            self.model_optimizer.zero_grad()
            (loss + e_attn_losses + e_ae_losses + d_attn_losses + d_ae_losses).backward()
            # for name, parameter in self.model.named_parameters():  # TODO test
            #     if parameter.grad is not None:
            #         print(name)
            # for name, parameter in self.compress_latents.named_parameters():  # TODO test
            #     if parameter.grad is not None:
            #         print(name)
            # for name, parameter in self.decompress_latents.named_parameters():  # TODO test
            #     if parameter.grad is not None:
            #         print(name)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.compress_latents.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.decompress_latents.parameters(), 0.1)
            self.model_optimizer.optimize()  # TODO add cosine decay and decreasing optimization updating

        losses = (loss.item(), e_attn_losses.item(), e_ae_losses.item(), d_attn_losses.item(), d_ae_losses.item(),
                  *loss_items)
        # # TODO adversarial autoencoder
        # self.model.zero_grad()
        # self.discriminator.zero_grad()
        # eps = 1e-15
        # # TODO Train discriminator
        # self.model.eval()
        # latents = []
        # for bar in bars:
        #     latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.model.encode(bar, e_mems, e_cmems)
        #     e_attn_losses.append(e_attn_loss)
        #     e_ae_losses.append(e_ae_loss)
        #     latents.append(latent)
        # latents = torch.stack(latents)
        # latents = self.compress_latents(latents)
        # z_real_gauss = Variable(torch.randn_like(latents)).to(self.device)  # normal distribution
        # d_real_gauss = self.discriminator(z_real_gauss)
        # d_fake_gauss = self.discriminator(latents)
        # d_loss = -torch.mean(torch.log(d_real_gauss + eps) + torch.log(1 - d_fake_gauss + eps))
        # d_loss.backward()
        # self.optim_D.step()
        # # TODO Train generator
        # self.model.train()
        # latents = []
        # for bar in bars:
        #     latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.model.encode(bar, e_mems, e_cmems)
        #     e_attn_losses.append(e_attn_loss)
        #     e_ae_losses.append(e_ae_loss)
        #     latents.append(latent)
        # latents = torch.stack(latents)
        # d_fake_gauss = self.D_net_gauss(latents)
        # g_loss = -torch.mean(torch.log(d_fake_gauss + eps))
        # g_loss.backward()
        # self.optim_Q_gen.step()
        # losses = (loss.item(), e_attn_losses.item(), e_ae_losses.item(), d_attn_losses.item(), d_ae_losses.item(),
        #           *loss_items, d_loss.item(), g_loss.item())
        return losses

    def log_to_wandb(self, losses):
        """
        It logs to wandb the losses w.r.t. the model mode
        :param losses: Tuple with the 9 losses
        :return: Nothing
        """
        mode = "train/" if self.model.training else "eval/"
        wandb.log({mode + "loss": losses[0],
                   mode + "encoder attention loss": losses[1],
                   mode + "encoder autoencoder loss": losses[2],
                   mode + "decoder attention loss": losses[3],
                   mode + "decoder autoencoder loss": losses[4],
                   mode + "drums loss": losses[5],
                   mode + "bass loss": losses[6],
                   mode + "guitar loss": losses[7],
                   mode + "strings loss": losses[8]})
        # mode + "recon loss": losses[9],
        # mode + "discriminator loss": losses[10],
        # mode + "generator loss": losses[11]})

    def reconstruct(self, model, original_song, note_manager=None):
        """
        :param model: The trained model
        :param original_song: Tensor with shape (n_bars, n_batch, timesteps)
        :param note_manager: NoteManager class to reconstruct the song
        :return: Music, Music: original Music and reconstructed Music
        """
        model.to(config["train"]["device"])
        model.eval()
        src = np.swapaxes(original_song, 0, 2)  # swap batch, tracks -> tracks, batch
        src = torch.LongTensor(src.long()).to(config["train"]["device"])
        e_mems, e_cmems, d_mems, d_cmems = self.get_memories()
        outs = []
        for bar in src:
            n_tokens = np.count_nonzero(bar.cpu())
            if n_tokens == 0:
                continue
            latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = model.encode(bar, e_mems, e_cmems)
            out, d_mems, d_cmems, d_attn_loss, d_ae_loss = model.decode(latent, bar, d_mems, d_cmems)
            out = torch.max(out, dim=-2).indices
            outs.append(out)
        outs = torch.stack(outs)
        src = src.cpu()
        outs = outs.cpu()
        outs = outs.transpose(0, 1)  # invert bar and batch
        outs = outs.transpose(0, -1)  # invert batch and instruments
        outs = outs[:, :, :, 0]  # take first song of batch
        src = src.transpose(0, 1)
        src = src.transpose(-2, -1)
        src = src[:, :, :, 0]
        original = note_manager.reconstruct_music(src.reshape(4, -1).numpy())
        reconstructed = note_manager.reconstruct_music(outs.reshape(4, -1).numpy())  # flat bars
        reconstructed = note_manager.cut_song(reconstructed, original.get_end_time())  # cut reconstructed song length
        return original, reconstructed

    def generate(self, iterations=10):
        """
        It generates a new song using the aae with a normal distributed vector.
        :param iterations: Number of latents to consider and so bars to generate
        :return: Muspy generated song
        """
        # z = Variable(torch.randn((self.batch_size, self.z_dim))).cuda()
        # latents = self.model.forward(z)
        # latents = latents.reshape(self.batch_size, self.max_bars, self.model.d_model)
        # latents = latents.transpose(0, 1)
        # _, _, d_mems, d_cmems = self.get_memories()
        # # 4 x 2 x 200
        # first = True
        # self.model.eval()
        # bars = []
        # latents = latents[:iterations, ...]
        # for latent in tqdm(latents, leave=True, position=0, desc="Generating track"):
        #     if first:
        #         bar = torch.LongTensor(4, self.batch_size, 1)  # n_track x n_batch x tokens
        #         bar[:, :, 0] = config["tokens"]["sos_token"]
        #         first = False
        #     else:
        #         bar = torch.LongTensor(4, self.batch_size, 1)
        #         bar[:, :, 0] = config["tokens"]["sob_token"]
        #     bar = bar.to(self.device)
        #     for i in range(max_bar_length-1):  # we start with sos token, so we stop une step before size
        #         to_pad = max_bar_length - bar.shape[-1]
        #         bar = F.pad(bar, (0, to_pad), value=config["tokens"]["pad_token"])
        #
        #         out, _, _, _, _ = self.model.decode(latent, bar, d_mems, d_cmems)
        #         out = torch.max(out, dim=-2).indices
        #         out = out.transpose(1, 2)
        #         out = out.transpose(0, 1)
        #         bar = torch.cat((bar[:, :, :(i+1)], out[:, :, 0].unsqueeze(-1)), dim=-1)
        #     out, d_mems, d_cmems, _, _ = self.model.decode(latent, bar, d_mems, d_cmems)
        #     bars.append(bar)
        # bars = torch.stack(bars)
        # note_manager = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])
        # bars = bars.transpose(0, 2)  # invert bars and batches
        # song = bars[0, :, :, :]  # take just first song
        # song = torch.reshape(song, (4, -1)).cpu().numpy()
        # song = note_manager.reconstruct_music(song)
        # return song
        pass

    def train(self):
        # Create checkpoint folder
        timestamp = str(datetime.now())
        timestamp = timestamp[:timestamp.index('.')]
        timestamp = timestamp.replace(' ', '_').replace(':', '-')
        self.save_path = timestamp
        os.mkdir(self.save_path)
        # Models
        self.model.to(config["train"]["device"])
        # Optimizers

        # gen_lr = 0.0001
        # reg_lr = 0.00005
        # self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=reg_lr)
        # self.optim_generator = torch.optim.Adam(self.model.encoder.parameters(), lr=reg_lr)
        # Loss
        criterion = LabelSmoothing(size=config["tokens"]["vocab_size"],
                                   padding_idx=config["tokens"]["pad_token"],
                                   smoothing=config["train"]["label_smoothing"],
                                   device=config["train"]["device"])
        criterion.to(config["train"]["device"])
        self.loss_computer = SimpleLossCompute(criterion)
        # Dataset
        dataset = SongIterator(dataset_path=config["paths"]["dataset_path"],
                               test_size=config["train"]["test_size"],
                               batch_size=config["train"]["batch_size"],
                               n_workers=config["train"]["n_workers"])
        tr_loader, ts_loader = dataset.get_loaders()
        # Wandb
        wandb.login()
        wandb.init(project="MusAE", config=self.settings, name="remote" if remote else "local" + ' ' + self.save_path)
        wandb.watch(self.model)
        # Train
        self.model.train()
        desc = "Train epoch " + str(self.epoch) + ", mb " + str(0)
        train_progress = tqdm(total=config["train"]["mb_before_eval"], position=0, leave=True, desc=desc)
        trained = False
        it_counter = 0
        best_ts_loss = float("inf")
        # best_aae_disc_loss = float("inf")
        # best_aae_gen_loss = float("inf")
        for self.epoch in range(config["train"]["n_epochs"]):  # repeat for each epoch
            for it, src in enumerate(tr_loader):  # repeat for each mini-batch
                if it % config["train"]["mb_before_eval"] == 0 and trained and False:  # TODO remove false
                    train_progress.close()
                    ts_losses = []
                    self.model.eval()
                    desc = "Eval epoch " + str(self.epoch) + ", mb " + str(it)
                    test = None
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
                    # if final[0] < 1:  # song are well reconstructed, we can check the aae model
                    #     if final[9] < best_aae_recon_loss and final[10] < best_aae_disc_loss and \
                    #             final[11] < best_aae_gen_loss:
                    #         print("Saving best aae model in " + os.path.join(self.save_path, "aae"))
                    #         for filename in glob.glob(os.path.join(self.save_path, "aae" + '*')):
                    #             os.remove(filename)
                    #         best_aae_recon_loss = final[9]
                    #         best_aae_disc_loss = final[10]
                    #         best_aae_gen_loss = final[11]
                    #         torch.save(self.Q_net, "aae_encoder")
                    #         torch.save(self.P_net, "aae_decoder")
                    #         torch.save(self.D_net_gauss, "aae_discriminator")
                    #         print("aae model saved")
                    if final[0] < best_ts_loss:
                        new_model = os.path.join(self.save_path, self.model_name + '_' + str(self.epoch) + '.pt')
                        print("Saving best model in " + new_model + ", DO NOT INTERRUPT")
                        for filename in glob.glob(os.path.join(self.save_path, self.model_name + '*')):
                            os.remove(filename)
                        best_ts_loss = final[0]
                        torch.save(self.model, new_model)
                        print("Model saved")
                    self.log_to_wandb(final)
                    # generated = self.generate(iterations=config["train"]["generated_iterations"])  # generation
                    note_manager = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])
                    original, reconstructed = self.reconstruct(self.model, test, note_manager)
                    prefix = "epoch_" + str(self.epoch) + "_mb_" + str(it)
                    original.write_midi(os.path.join(wandb.run.dir, prefix + "_original.mid"))
                    reconstructed.write_midi(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"))
                    # generated.write_midi(os.path.join(wandb.run.dir, prefix+"_generated.mid"))
                    midi_to_wav(os.path.join(wandb.run.dir, prefix + "_original.mid"),
                                os.path.join(wandb.run.dir, prefix + "_original.wav"))
                    midi_to_wav(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"),
                                os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"))
                    # midi_to_wav(os.path.join(wandb.run.dir, prefix + "_generated.mid"),
                    #             os.path.join(wandb.run.dir, prefix + "_generated.wav"))
                    wandb.log({str(it_counter): [
                        wandb.Audio(os.path.join(wandb.run.dir, prefix + "_original.wav"),
                                    caption="original", sample_rate=32),
                        wandb.Audio(os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"),
                                    caption="reconstructed", sample_rate=32),
                        # wandb.Audio(os.path.join(wandb.run.dir, prefix + "_generated.wav"),
                        #             caption="generated", sample_rate=32)
                    ]})
                    it_counter += 1
                    # wandb.save(os.path.join(wandb.run.dir, prefix + "_original.mid"))
                    # wandb.save(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"))
                    self.model.train()
                    desc = "Train epoch " + str(self.epoch) + ", mb " + str(it)
                    train_progress = tqdm(total=config["train"]["mb_before_eval"], position=0, leave=True, desc=desc)
                tr_losses = self.run_mb(src)
                self.log_to_wandb(tr_losses)
                train_progress.update()
                trained = True
