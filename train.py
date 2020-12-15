import matplotlib.pyplot as plt
import os
import torch
from datetime import datetime
from tqdm.auto import tqdm
from compressive_transformer import TransformerAutoencoder
from config import config, set_freer_gpu
from iterate_dataset import SongIterator
from optimizer import NoamOpt
from label_smoother import LabelSmoothing
from loss_computer import SimpleLossCompute
import numpy as np
from create_dataset import NoteRepresentationManager
import shutil
import sys
import glob
import wandb


# TO COPY: scp -r C:\Users\berti\PycharmProjects\MusAE\*.py berti@131.114.137.168:MusAE
# TO CONNECT: ssh berti@131.114.137.168
# TO ATTACH TO TMUX: tmux attach -t Training
# TO RESIZE TMUX: tmux attach -d -t Training
# TO SWITCH WINDOW ctrl+b 0-1-2
# TO SEE SESSION: tmux ls
# TO DETACH ctrl+b d
# TO VISUALIZE GPUs STATUS: nvidia-smi
# TO GET RESULTS: scp -r berti@131.114.137.168:MusAE/training* C:\Users\berti\PycharmProjects\MusAE\remote_results

class Trainer:
    def __init__(self, save_path=None, device=None, dataset_path=None, test_size=None,
                 batch_size=None, n_workers=None, vocab_size=None, n_epochs=None, model_name="checkpoint",
                 plot_name="plot", model=None, max_bars=None, label_smoothing=None, config=config):
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

    def plot(self, tr, ts, tr_aux, ts_aux, tr_ae, ts_ae, plot_path):
        if self.epoch == 0:
            plt.figure()
        plt.subplot(311)  # 3 rows and 1 column, first index
        plt.plot(range(self.epoch + 1), tr, c="r", label="Training loss")
        plt.plot(range(self.epoch + 1), ts, c="b", label="Testing loss")
        plt.subplot(312)  # 3 rows and 1 column, second index
        plt.plot(range(self.epoch + 1), tr_aux, c="r", label="Training aux loss")
        plt.plot(range(self.epoch + 1), ts_aux, c="b", label="Testing aux loss")
        plt.subplot(313)  # 3 rows and 1 column, third index
        plt.plot(range(self.epoch + 1), tr_ae, c="r", label="Training auto-encoding loss")
        plt.plot(range(self.epoch + 1), ts_ae, c="b", label="Testing auto-encoding loss")
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()

    def run_epoch(self, loader):
        total_loss = 0
        total_aux_loss = 0
        total_ae_loss = 0
        i = 0
        description = ("Train" if self.model.training else "Eval ") + " epoch " + str(self.epoch)
        # print(description)
        for src in tqdm(loader, desc=description, leave=True, position=0):
        # length = len(loader)
        # for count, src in enumerate(loader):
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
                latent, e_mems, e_cmems, e_attn_loss, e_ae_loss = self.model.encode(bar, e_mems, e_cmems)
                out, d_mems, d_cmems, d_attn_loss, d_ae_loss = self.model.decode(latent, bar, d_mems, d_cmems)

                loss, loss_items = self.loss_computer(out, bar[:, :, :], n_tokens)  # TODO skip first elem of each bars>
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    (loss + e_attn_loss + e_ae_loss + d_attn_loss + d_attn_loss).backward()
                    self.optimizer.step()

                loss_drums, loss_bass, loss_guitar, loss_strings = loss_items
                drums_losses.append(loss_drums)
                bass_losses.append(loss_bass)
                guitar_losses.append(loss_guitar)
                strings_losses.append(loss_guitar)

                losses.append(loss.item())
                e_attn_losses.append(e_attn_loss.item())
                e_ae_losses.append(e_ae_loss.item())
                d_attn_losses.append(d_attn_loss.item())
                d_ae_losses.append(d_ae_loss.item())

            loss_avg = sum(losses)/len(losses)
            e_attn_loss_avg = sum(e_attn_losses)/len(e_attn_losses)
            e_ae_loss_avg = sum(e_ae_losses)/len(e_ae_losses)
            d_attn_loss_avg = sum(d_attn_losses)/len(d_attn_losses)
            d_ae_loss_avg = sum(d_ae_losses)/len(d_ae_losses)
            drums_loss_avg = sum(drums_losses)/len(drums_losses)
            bass_loss_avg = sum(bass_losses)/len(bass_losses)
            guitar_loss_avg = sum(guitar_losses)/len(guitar_losses)
            strings_loss_avg = sum(guitar_losses)/len(guitar_losses)

            mode = "train/" if self.model.training else "eval/"
            wandb.log({mode+"loss": loss_avg,
                       mode+"encoder attention loss": e_attn_loss_avg,
                       mode+"encoder autoencoder loss": e_ae_loss_avg,
                       mode+"decoder attention loss": d_attn_loss_avg,
                       mode+"decoder autoencoder loss": d_ae_loss_avg,
                       mode+"drums loss": drums_loss_avg,
                       mode+"bass loss": bass_loss_avg,
                       mode+"guitar loss": guitar_loss_avg,
                       mode+"strings loss": strings_loss_avg})


            # print("{:.1f}% Mini-batch: loss: {:.4f}, e_attn loss: {:.4f}, e_ae loss: {:.4f},"
            #       " d_attn loss: {:.4f}, d_ae loss: {:.4f}".format(
            #        (count/length)*100, loss_avg, e_attn_loss_avg, e_ae_loss_avg, d_attn_loss_avg, d_ae_loss_avg, end="\n"))

            total_loss += loss_avg
            total_aux_loss += (e_attn_loss_avg + d_attn_loss_avg) / 2
            total_ae_loss += (e_ae_loss_avg + d_ae_loss_avg) / 2

            i += 1
        if i == 0:
            exit("i is zero for some kind of mystery")
        return total_loss / i, total_aux_loss / i, total_ae_loss / i

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
        self.optimizer = torch.optim.Adam(self.model.parameters())  # TODO adjust following paper
        # Loss
        criterion = LabelSmoothing(size=self.vocab_size, padding_idx=self.model.pad_token,
                                   smoothing=self.label_smoothing, device=self.device)
        criterion.to(self.device)
        self.loss_computer = SimpleLossCompute(criterion)
        # tr_losses = []
        # ts_losses = []
        # tr_aux_losses = []
        # ts_aux_losses = []
        # tr_ae_losses = []
        # ts_ae_losses = []
        best_ts_loss = 1000000
        dataset = SongIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                               batch_size=self.batch_size, n_workers=self.n_workers)
        tr_loader, ts_loader = dataset.get_loaders()
        wandb.login()
        if os.getcwd() == 'C:\\Users\\berti\\PycharmProjects\\MusAE':
            execution = "local"
        else:
            execution = "remote"
        wandb.init(project="MusAE", config=self.config, name=execution+' '+self.save_path)
        config = wandb.config
        wandb.watch(self.model)
        # Train
        for self.epoch in range(self.n_epochs):
            # print("Epoch ", epoch, " over ", n_epochs)
            self.model.train()
            tr_loss, tr_aux_loss, tr_ae_loss = self.run_epoch(tr_loader)
            self.model.eval()
            ts_loss, ts_aux_loss, ts_ae_loss = self.run_epoch(ts_loader)
            # print("Epoch {}: TR loss: {:.4f}, TS loss: {:.4f}, Aux TR loss: {:.4f}, Aux TS loss: {:.2f}, "
            #       "AE TR loss: {:.4f}, AE TS loss: {:.4f}".format(
            #        self.epoch, tr_loss, ts_loss, tr_aux_loss, ts_aux_loss, tr_ae_loss, ts_ae_loss, end="\n"))
            # Save model if best and erase all others
            if ts_loss < best_ts_loss:
                new_model = os.path.join(self.save_path, self.model_name + '_' + str(self.epoch) + '.pt')
                print("Saving best model in " + new_model + ", DO NOT INTERRUPT")
                for filename in glob.glob(os.path.join(self.save_path, self.model_name+'*')):
                    os.remove(filename)
                best_ts_loss = ts_loss
                torch.save(self.model, new_model)
                print("Model saved")
            # # Plot learning curve and save it
            # tr_losses.append(tr_loss)
            # ts_losses.append(ts_loss)
            # tr_aux_losses.append(tr_aux_loss)
            # ts_aux_losses.append(ts_aux_loss)
            # tr_ae_losses.append(tr_ae_loss)
            # ts_ae_losses.append(ts_ae_loss)
            # plot_path = os.path.join(self.save_path, self.plot_name + '_' + str(self.epoch) + '.png')
            # self.plot(tr_losses, ts_losses, tr_aux_losses, ts_aux_losses,
            #           tr_ae_losses, ts_ae_losses, plot_path=plot_path)
            # # Remove old plot
            # if self.epoch > 0:
            #     os.remove(os.path.join(self.save_path, self.plot_name + '_' + str(self.epoch - 1) + '.png'))
            # Save example
            note_manager = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])
            original, reconstructed = self.generate(self.model, note_manager)
            original.write_midi(os.path.join(wandb.run.dir, "epoch_"+str(self.epoch)+"_original.mid"))
            reconstructed.write_midi(os.path.join(wandb.run.dir, "epoch_"+str(self.epoch)+"_reconstructed.mid"))
            wandb.save(os.path.join(wandb.run.dir, "epoch_"+str(self.epoch)+"_original.mid"))
            wandb.save(os.path.join(wandb.run.dir, "epoch_"+str(self.epoch)+"_reconstructed.mid"))

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

    def generate(self, model, note_manager=None):
        model.to(self.device)
        model.eval()
        _, ts_loader = SongIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                                    batch_size=self.batch_size, n_workers=self.n_workers).get_loaders()
        original_song = ts_loader.__iter__().next()
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
# self.optimizer = NoamOpt(config["model"]["d_model"], 1, 2000,
#                          torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
