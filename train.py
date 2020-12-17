import matplotlib.pyplot as plt
import os
import torch
from datetime import datetime
from tqdm.auto import tqdm
from compressive_transformer import TransformerAutoencoder
from config import config, set_freer_gpu
from iterate_dataset import SongIterator
from optimizer import CTOpt
from label_smoother import LabelSmoothing
from loss_computer import SimpleLossCompute
import numpy as np
from create_dataset import NoteRepresentationManager
import shutil
import sys
import glob
import wandb
from midi_converter import midi_to_wav


# TO COPY: scp -r C:\Users\berti\PycharmProjects\MusAE\*.py berti@131.114.137.168:MusAE
# TO CONNECT: ssh berti@131.114.137.168
# TO ATTACH TO TMUX: tmux attach -t Training
# TO RESIZE TMUX: tmux attach -d -t Training
# TO SWITCH WINDOW ctrl+b 0-1-2
# TO SEE SESSION: tmux ls
# TO DETACH ctrl+b d
# TO VISUALIZE GPUs STATUS: nvidia-smi
# TO GET RESULTS: scp -r berti@131.114.137.168:MusAE/2020* C:\Users\berti\PycharmProjects\MusAE\remote_results

class Trainer:
    def __init__(self, save_path=None, device=None, dataset_path=None, test_size=None,
                 batch_size=None, n_workers=None, vocab_size=None, n_epochs=None, model_name="checkpoint",
                 plot_name="plot", model=None, max_bars=None, label_smoothing=None, config=config,
                 mb_before_eval=None):
        self.epoch = 0
        self.save_path = save_path
        self.model = None
        self.loss_computer = None
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
        self.optimizer = None
        self.mb_before_eval = mb_before_eval

    def run_mb(self, src):
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

        # for bar in tqdm(bars, leave=True, position=0):
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
            out, d_mems, d_cmems, d_attn_loss, d_ae_loss = self.model.decode(latent, bar, d_mems, d_cmems)

            loss, loss_items = self.loss_computer(out, bar[:, :, 1:], norm)  # TODO skip first elem of each bars>
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
                  bass_loss_avg, guitar_loss_avg, strings_loss_avg)
        return losses

    def log_to_wandb(self, losses):
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

    def train(self):
        if not self.save_path:  # if no save path is defined (to default it is not)
            timestamp = str(datetime.now())
            timestamp = timestamp[:timestamp.index('.')]
            timestamp = timestamp.replace(' ', '_').replace(':', '-')
            # self.save_path = 'training_' + timestamp
            self.save_path = timestamp

        os.mkdir(self.save_path)
        # Model
        self.model.to(self.device)
        # Optimizer
        self.optimizer = CTOpt(torch.optim.Adam(self.model.parameters()), 1000, (1e-6, 1e-4))
        # Loss
        criterion = LabelSmoothing(size=self.vocab_size, padding_idx=self.model.pad_token,
                                   smoothing=self.label_smoothing, device=self.device)
        criterion.to(self.device)
        self.loss_computer = SimpleLossCompute(criterion)
        best_ts_loss = 1000000
        dataset = SongIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                               batch_size=self.batch_size, n_workers=self.n_workers)
        tr_loader, ts_loader = dataset.get_loaders()
        wandb.login()
        if os.getcwd() == 'C:\\Users\\berti\\PycharmProjects\\MusAE':
            execution = "local"
        else:
            execution = "remote"
        wandb.init(project="MusAE", config=self.config, name=execution + ' ' + self.save_path)
        config = wandb.config
        wandb.watch(self.model)
        # Train
        self.model.train()
        desc = "Train epoch " + str(self.epoch) + ", mb " + str(0)
        train_progress = tqdm(total=self.mb_before_eval, position=0, leave=True, desc=desc)
        trained = False
        it_counter = 0
        for self.epoch in range(self.n_epochs):  # repeat for each epoch
            # print("Epoch ", self.epoch)
            for it, src in enumerate(tr_loader):  # repeat for each mini-batch
                if it % self.mb_before_eval == 0 and trained:
                    train_progress.close()
                    ts_losses = []
                    self.model.eval()
                    desc = "Eval epoch " + str(self.epoch) + ", mb " + str(it)
                    test = None
                    for test in tqdm(ts_loader, position=0, leave=True, desc=desc):
                        ts_loss = self.run_mb(test)
                        ts_losses.append(ts_loss)
                    final = ()
                    for i in range(len(ts_losses[0])):
                        aux = []
                        for loss in ts_losses:
                            aux.append(loss[i])
                        avg = sum(aux) / len(aux)
                        final = final + (avg,)
                    self.log_to_wandb(final)
                    if final[0] < best_ts_loss:
                        new_model = os.path.join(self.save_path, self.model_name + '_' + str(self.epoch) + '.pt')
                        print("Saving best model in " + new_model + ", DO NOT INTERRUPT")
                        for filename in glob.glob(os.path.join(self.save_path, self.model_name + '*')):
                            os.remove(filename)
                        best_ts_loss = final[0]
                        torch.save(self.model, new_model)
                        print("Model saved")
                    note_manager = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])
                    original, reconstructed = self.generate(self.model, test, note_manager)  # TODO ATTENTION
                    prefix = "epoch_" + str(self.epoch) + "_mb_" + str(it)
                    original.write_midi(os.path.join(wandb.run.dir, prefix + "_original.mid"))  # TODO write as wav
                    reconstructed.write_midi(os.path.join(wandb.run.dir, prefix+"_reconstructed.mid"))
                    midi_to_wav(os.path.join(wandb.run.dir, prefix + "_original.mid"),
                                os.path.join(wandb.run.dir, prefix + "_original.wav"))
                    midi_to_wav(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"),
                                os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"))
                    wandb.log({str(it_counter):
                                   [wandb.Audio(os.path.join(wandb.run.dir, prefix + "_original.wav"),
                                                caption="original", sample_rate=32),
                                    wandb.Audio(os.path.join(wandb.run.dir, prefix + "_reconstructed.wav"),
                                                caption="reconstructed", sample_rate=32)
                                    ]})
                    it_counter += 1
                    # wandb.save(os.path.join(wandb.run.dir, prefix + "_original.mid"))
                    # wandb.save(os.path.join(wandb.run.dir, prefix + "_reconstructed.mid"))
                    self.model.train()
                    desc = "Train epoch " + str(self.epoch) + ", mb " + str(it)
                    train_progress = tqdm(total=10, position=0, leave=True, desc=desc)
                tr_losses = self.run_mb(src)
                self.log_to_wandb(tr_losses)
                train_progress.update()
                trained = True

    def get_memories(self):
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

    def generate(self, model, original_song, note_manager=None):
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


if __name__ == "__main__":
    set_freer_gpu()
    notes = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])

    shutil.rmtree(config["paths"]["dataset_path"], ignore_errors=True)
    notes.convert_dataset()

    m = TransformerAutoencoder(**config["model"])

    trainer = Trainer(model=m,
                      dataset_path=config["paths"]["dataset_path"],
                      **config["train"],
                      config=config,
                      )
    trainer.train()

# TODO remove test
# for parameter in self.model.named_parameters():
# print(parameter[0], " ", parameter[1].shape)

# Optimizer
# self.opt = NoamOpt(config["model"]["d_model"], 1, 2000,
#                          torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    # def plot(self, tr, ts, tr_aux, ts_aux, tr_ae, ts_ae, plot_path):
    #     if self.epoch == 0:
    #         plt.figure()
    #     plt.subplot(311)  # 3 rows and 1 column, first index
    #     plt.plot(range(self.epoch + 1), tr, c="r", label="Training loss")
    #     plt.plot(range(self.epoch + 1), ts, c="b", label="Testing loss")
    #     plt.subplot(312)  # 3 rows and 1 column, second index
    #     plt.plot(range(self.epoch + 1), tr_aux, c="r", label="Training aux loss")
    #     plt.plot(range(self.epoch + 1), ts_aux, c="b", label="Testing aux loss")
    #     plt.subplot(313)  # 3 rows and 1 column, third index
    #     plt.plot(range(self.epoch + 1), tr_ae, c="r", label="Training auto-encoding loss")
    #     plt.plot(range(self.epoch + 1), ts_ae, c="b", label="Testing auto-encoding loss")
    #     plt.legend()
    #     plt.savefig(plot_path)
    #     plt.clf()