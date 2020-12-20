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
import torch.nn.functional as F
from aae import Q_net, P_net, D_net_gauss
from torch.autograd import Variable
from config import max_bar_length


class Trainer:
    def __init__(self, save_path=None, device=None, dataset_path=None, test_size=None,
                 batch_size=None, n_workers=None, vocab_size=None, n_epochs=None, model_name="checkpoint",
                 plot_name="plot", model=None, max_bars=None, label_smoothing=None, config=config,
                 mb_before_eval=None, warmup_steps=None, lr_min=None, lr_max=None, z_dim=None, mid_dim=None):
        self.save_path = save_path
        self.device = device
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.vocab_size = vocab_size
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.plot_name = plot_name
        self.model = model
        self.max_bars = max_bars
        self.label_smoothing = label_smoothing
        self.config = config
        self.mb_before_eval = mb_before_eval
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.epoch = 0
        self.model = model
        self.loss_computer = None
        self.optimizer = None
        # Adversarial part
        self.z_dim = z_dim
        self.mid_dim = mid_dim
        self.Q_net = None
        self.P_net = None
        self.D_net_gauss = None
        self.optim_P = None
        self.optim_Q_enc = None
        self.optim_Q_gen = None
        self.optim_D = None

    def get_memories(self):
        """
        :return: 4 empty memories with dimension taken from attributes of the self class
        """
        a = self.model.n_tracks
        b = self.model.layers
        c = self.batch_size
        d = 0  # initial memories dimension
        e = self.model.d_model
        e_mems = torch.empty(a, b, c, d, e, dtype=torch.float32, device=self.device)
        e_cmems = torch.empty(a, b, c, d, e, dtype=torch.float32, device=self.device)
        d_mems = torch.empty(a, b, c, d, e, dtype=torch.float32, device=self.device)
        d_cmems = torch.empty(a, b, c, d, e, dtype=torch.float32, device=self.device)
        return e_mems, e_cmems, d_mems, d_cmems

    def run_mb(self, src):
        """
        It runs a mini-batch and return loss, if the model is in training it does also optimization
        :param src: Tensor with shape (n_batch, time_steps)
        :return: Tuple with total_loss, auxiliary_losses and instruments losses
        """
        src = np.swapaxes(src, 0, 2)  # swap batch, tracks -> tracks, batch
        bars = torch.LongTensor(src.long()).to(self.device)
        e_mems, e_cmems, d_mems, d_cmems = self.get_memories()
        losses = []
        e_attn_losses = []
        e_ae_losses = []
        d_attn_losses = []
        d_ae_losses = []
        drums_losses = []
        bass_losses = []
        guitar_losses = []
        strings_losses = []
        latents = []
        for bar in bars:
            n_tokens = np.count_nonzero(bar.cpu())
            if n_tokens == 0:
                continue
            n_tokens_drums = np.count_nonzero(bar.cpu()[0, :, :])
            n_tokens_bass = np.count_nonzero(bar.cpu()[1, :, :])
            n_tokens_guitar = np.count_nonzero(bar.cpu()[2, :, :])
            n_tokens_strings = np.count_nonzero(bar.cpu()[3, :, :])
            norm = (n_tokens, n_tokens_drums, n_tokens_bass, n_tokens_guitar, n_tokens_strings)

            latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.model.encode(bar, e_mems, e_cmems)
            latents.append(latent.data)  # aae decoder output has sigmoid, so our latents has sigmoid too
            out, d_mems, d_cmems, d_attn_loss, d_ae_loss = self.model.decode(latent, bar, d_mems, d_cmems)

            loss, loss_items = self.loss_computer(out, bar[:, :, 1:], norm)
            if self.model.training:
                self.optimizer.zero_grad()
                (loss + e_attn_loss + e_ae_loss + d_attn_loss + d_attn_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.optimize()  # TODO add cosine decay and decreasing optimization updating

            loss_drums, loss_bass, loss_guitar, loss_strings = loss_items
            if loss_drums != -1:  # if this instrument was not empty in this bar
                drums_losses.append(loss_drums)
            if loss_bass != -1:
                bass_losses.append(loss_bass)
            if loss_guitar != -1:
                guitar_losses.append(loss_guitar)
            if loss_strings != -1:
                strings_losses.append(loss_guitar)

            losses.append(loss.item())
            e_attn_losses.append(e_attn_loss.item())
            e_ae_losses.append(e_ae_loss.item())
            d_attn_losses.append(d_attn_loss.item())
            d_ae_losses.append(d_ae_loss.item())

        # TODO train variational autoencoder for latents compression
        EPS = 1e-15
        latents = torch.stack(latents)
        to_pad = self.max_bars - latents.shape[0]
        latents = F.pad(latents, (0, 0, 0, 0, 0, to_pad), value=0.)
        latents = latents.transpose(0, 1)
        latents = torch.reshape(latents, (self.batch_size, -1)).to(self.device)
        # Reconstruction loss: optimize encoder Q and decoder P to reconstruct input
        self.P_net.zero_grad()
        self.Q_net.zero_grad()
        self.D_net_gauss.zero_grad()
        z = self.Q_net(latents)
        x = self.P_net(z)
        # recon_loss = F.binary_cross_entropy(x+EPS, latents+EPS)  TODO mse or binary cross entropy?
        recon_loss = F.mse_loss(x + EPS, latents + EPS)
        # recon_loss = recon_loss / (self.model.d_model*self.max_bars)
        recon_loss.backward()
        self.optim_P.step()
        self.optim_Q_enc.step()
        # Discriminator: optimize discriminator to recognize normal distribution as valid
        self.Q_net.eval()
        z_real_gauss = Variable(torch.randn_like(z)).cuda()  # normal distribution
        D_real_gauss = self.D_net_gauss(z_real_gauss)
        z_fake_gauss = self.Q_net(latents)
        D_fake_gauss = self.D_net_gauss(z_fake_gauss)
        D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
        D_loss.backward()
        self.optim_D.step()
        # Generator
        self.Q_net.train()
        z_fake_gauss = self.Q_net(latents)
        D_fake_gauss = self.D_net_gauss(z_fake_gauss)
        G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
        G_loss.backward()
        self.optim_Q_gen.step()
        # Avg losses
        loss_avg = sum(losses) / len(losses)
        e_attn_loss_avg = sum(e_attn_losses) / len(e_attn_losses)
        e_ae_loss_avg = sum(e_ae_losses) / len(e_ae_losses)
        d_attn_loss_avg = sum(d_attn_losses) / len(d_attn_losses)
        d_ae_loss_avg = sum(d_ae_losses) / len(d_ae_losses)
        drums_loss_avg = sum(drums_losses) / len(drums_losses)
        bass_loss_avg = sum(bass_losses) / len(bass_losses)
        guitar_loss_avg = sum(guitar_losses) / len(guitar_losses)
        strings_loss_avg = sum(guitar_losses) / len(guitar_losses)
        losses = (loss_avg, e_attn_loss_avg, e_ae_loss_avg, d_attn_loss_avg, d_ae_loss_avg, drums_loss_avg,
                  bass_loss_avg, guitar_loss_avg, strings_loss_avg, recon_loss, D_loss, G_loss)
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
                   mode + "strings loss": losses[8],
                   mode + "recon loss": losses[9],
                   mode + "discriminator loss": losses[10],
                   mode + "generator loss": losses[11]})

    def reconstruct(self, model, original_song, note_manager=None):
        """
        :param model: The trained model
        :param original_song: Tensor with shape (n_bars, n_batch, timesteps)
        :param note_manager: NoteManager class to reconstruct the song
        :return: Music, Music: original Music and reconstructed Music
        """
        model.to(self.device)
        model.eval()
        src = np.swapaxes(original_song, 0, 2)  # swap batch, tracks -> tracks, batch
        src = torch.LongTensor(src.long()).to(self.device)
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
        z = Variable(torch.randn((self.batch_size, self.z_dim))).cuda()
        latents = self.P_net(z)
        latents = latents.reshape(self.batch_size, self.max_bars, self.model.d_model)
        latents = latents.transpose(0, 1)
        _, _, d_mems, d_cmems = self.get_memories()
        # 4 x 2 x 200
        first = True
        self.model.eval()
        bars = []
        latents = latents[:iterations, ...]
        for latent in tqdm(latents, leave=True, position=0, desc="Generating track"):
            if first:
                bar = torch.LongTensor(4, self.batch_size, 1)  # n_track x n_batch x tokens
                bar[:, :, 0] = config["tokens"]["sos_token"]
                first = False
            else:
                bar = torch.LongTensor(4, self.batch_size, 1)
                bar[:, :, 0] = config["tokens"]["sob_token"]
            bar = bar.to(self.device)
            for i in range(max_bar_length-1):  # we start with sos token, so we stop une step before size
                to_pad = max_bar_length - bar.shape[-1]
                bar = F.pad(bar, (0, to_pad), value=config["tokens"]["pad_token"])

                out, _, _, _, _ = self.model.decode(latent, bar, d_mems, d_cmems)
                out = torch.max(out, dim=-2).indices
                out = out.transpose(1, 2)
                out = out.transpose(0, 1)
                bar = torch.cat((bar[:, :, :(i+1)], out[:, :, 0].unsqueeze(-1)), dim=-1)
            out, d_mems, d_cmems, _, _ = self.model.decode(latent, bar, d_mems, d_cmems)
            bars.append(bar)
        bars = torch.stack(bars)
        note_manager = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])
        bars = bars.transpose(0, 2)  # invert bars and batches
        song = bars[0, :, :, :]  # take just first song
        song = torch.reshape(song, (4, -1)).cpu().numpy()
        song = note_manager.reconstruct_music(song)
        return song

    def train(self):
        # Create checkpoint folder
        timestamp = str(datetime.now())
        timestamp = timestamp[:timestamp.index('.')]
        timestamp = timestamp.replace(' ', '_').replace(':', '-')
        self.save_path = timestamp
        os.mkdir(self.save_path)
        # Models
        self.model.to(self.device)
        self.P_net = P_net(self.max_bars*self.model.d_model, self.mid_dim, self.z_dim).to(self.device)
        self.Q_net = Q_net(self.max_bars*self.model.d_model, self.mid_dim, self.z_dim).to(self.device)
        self.D_net_gauss = D_net_gauss(self.max_bars*self.model.d_model, self.z_dim).to(self.device)
        # Optimizers
        self.optimizer = CTOpt(torch.optim.Adam(self.model.parameters()), self.warmup_steps,
                               (self.lr_min, self.lr_max))  # TODO add other
        gen_lr = 0.0001
        reg_lr = 0.00005
        self.optim_P = torch.optim.Adam(self.P_net.parameters(), lr=gen_lr)
        self.optim_Q_enc = torch.optim.Adam(self.Q_net.parameters(), lr=gen_lr)
        self.optim_Q_gen = torch.optim.Adam(self.Q_net.parameters(), lr=reg_lr)
        self.optim_D = torch.optim.Adam(self.D_net_gauss.parameters(), lr=reg_lr)
        # Loss
        criterion = LabelSmoothing(size=self.vocab_size, padding_idx=self.model.pad_token,
                                   smoothing=self.label_smoothing, device=self.device)
        criterion.to(self.device)
        self.loss_computer = SimpleLossCompute(criterion)
        # Dataset
        dataset = SongIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                               batch_size=self.batch_size, n_workers=self.n_workers)
        tr_loader, ts_loader = dataset.get_loaders()
        # Wandb
        wandb.login()
        wandb.init(project="MusAE", config=self.config, name="remote" if remote else "local" + ' ' + self.save_path)
        wandb.watch(self.model)
        # Train
        self.model.train()
        desc = "Train epoch " + str(self.epoch) + ", mb " + str(0)
        train_progress = tqdm(total=self.mb_before_eval, position=0, leave=True, desc=desc)
        trained = False
        it_counter = 0
        best_ts_loss = float("inf")
        best_aae_recon_loss = float("inf")
        best_aae_disc_loss = float("inf")
        best_aae_gen_loss = float("inf")
        for self.epoch in range(self.n_epochs):  # repeat for each epoch
            for it, src in enumerate(tr_loader):  # repeat for each mini-batch
                if it % self.mb_before_eval == 0 and trained:
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
                    if final[0] < 1:  # song are well reconstructed, we can check the aae model
                        if final[9] < best_aae_recon_loss and final[10] < best_aae_disc_loss and \
                                final[11] < best_aae_gen_loss:
                            print("Saving best aae model in " + os.path.join(self.save_path, "aae"))
                            for filename in glob.glob(os.path.join(self.save_path, "aae" + '*')):
                                os.remove(filename)
                            best_aae_recon_loss = final[9]
                            best_aae_disc_loss = final[10]
                            best_aae_gen_loss = final[11]
                            torch.save(self.Q_net, "aae_encoder")
                            torch.save(self.P_net, "aae_decoder")
                            torch.save(self.D_net_gauss, "aae_discriminator")
                            print("aae model saved")
                    if final[0] < best_ts_loss:
                        new_model = os.path.join(self.save_path, self.model_name + '_' + str(self.epoch) + '.pt')
                        print("Saving best model in " + new_model + ", DO NOT INTERRUPT")
                        for filename in glob.glob(os.path.join(self.save_path, self.model_name + '*')):
                            os.remove(filename)
                        best_ts_loss = final[0]
                        torch.save(self.model, new_model)
                        print("Model saved")
                    self.log_to_wandb(final)
                    generated = self.generate(iterations=10)  # generate new track
                    note_manager = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])
                    original, reconstructed = self.reconstruct(self.model, test, note_manager)
                    prefix = "epoch_" + str(self.epoch) + "_mb_" + str(it)
                    original.write_midi(os.path.join(wandb.run.dir, prefix + "_original.mid"))
                    reconstructed.write_midi(os.path.join(wandb.run.dir, prefix+"_reconstructed.mid"))
                    generated.write_midi(os.path.join(wandb.run.dir, prefix+"_generated.mid"))
                    midi_to_wav(os.path.join(wandb.run.dir, prefix + "_original.mid"),
                                os.path.join(wandb.run.dir, prefix + "_original.wav"))
                    midi_to_wav(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"),
                                os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"))
                    midi_to_wav(os.path.join(wandb.run.dir, prefix + "_generated.mid"),
                                os.path.join(wandb.run.dir, prefix + "_generated.wav"))
                    wandb.log({str(it_counter):
                                   [wandb.Audio(os.path.join(wandb.run.dir, prefix + "_original.wav"),
                                                caption="original", sample_rate=32),
                                    wandb.Audio(os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"),
                                                caption="reconstructed", sample_rate=32),
                                    wandb.Audio(os.path.join(wandb.run.dir, prefix + "_generated.wav"),
                                                caption="generated", sample_rate=32)
                                    ]})
                    it_counter += 1
                    # wandb.save(os.path.join(wandb.run.dir, prefix + "_original.mid"))
                    # wandb.save(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"))
                    self.model.train()
                    desc = "Train epoch " + str(self.epoch) + ", mb " + str(it)
                    train_progress = tqdm(total=self.mb_before_eval, position=0, leave=True, desc=desc)
                tr_losses = self.run_mb(src)
                self.log_to_wandb(tr_losses)
                train_progress.update()
                trained = True
