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


# TO COPY: scp -r C:\Users\berti\PycharmProjects\MusAE berti@131.114.137.168:
# TO CONNECT: ssh berti@131.114.137.168
# TO ATTACH TO TMUX: tmux attach -t Training
# TO SWITCH WINDOW ctrl+b 0-1-2
# TO SEE SESSION: tmux ls
# TO DETACH ctrl+b d
# TO VISUALIZE GPUs STATUS: nvidia-smi

class Trainer:
    def __init__(self, save_path=None, pad_token=None, device=None, dataset_path=None, test_size=None,
                 batch_size=None, n_workers=None, vocab_size=None, n_epochs=None, model_name="checkpoint",
                 plot_name="plot", model=None, max_bars=None, label_smoothing=None):
        self.epoch = 0
        self.save_path = save_path
        self.model = None
        self.loss_computer = None
        self.optimizer = None
        self.pad_token = pad_token
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

    def plot(self, tr, ts, tr_aux, ts_aux, plot_path):
        if self.epoch == 0:
            plt.figure()
        plt.subplot(211)  # 2 rows and 1 column, first index
        plt.plot(range(self.epoch + 1), tr, c="r", label="Training loss")
        plt.plot(range(self.epoch + 1), ts, c="b", label="Testing loss")
        plt.legend()
        plt.subplot(212)  # 2 rows and 1 column, second index
        plt.plot(range(self.epoch + 1), tr_aux, c="g", label="Training aux loss")
        plt.plot(range(self.epoch + 1), ts_aux, c="y", label="Testing aux loss")
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()

    def run_epoch(self, loader):
        total_loss = 0
        total_aux_loss = 0
        i = 0
        description = ("Train" if self.model.training else "Eval ") + " epoch " + str(self.epoch)
        for src in tqdm(loader, desc=description, leave=True, position=0):
            src = np.swapaxes(src, 0, 1)  # swap batch, tracks -> tracks, batch
            src = np.swapaxes(src, 1, 2)  # swap batch, bars -> bars, batch
            n_tokens = np.count_nonzero(src)
            src = torch.LongTensor(src.long()).to(self.device)
            if n_tokens == 0:  # All tracks are empty
                continue
            # Step
            out, aux_loss = self.model.forward(src)  # TODO why out is none on remote?
            # print("out: ", out)
            loss = self.loss_computer(out, src[:, :, :, 1:], n_tokens)  # skip first elem of each bars
            if self.optimizer is not None:
                self.optimizer.optimizer.zero_grad()
                (aux_loss + loss).backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_aux_loss += aux_loss.item()
            i += 1
        if i == 0:
            exit("i is zero for some kind of mystery")
        return total_loss / i, total_aux_loss / i

    def train(self):
        if not self.save_path:  # if no save path is defined (to default it is not)
            timestamp = str(datetime.now())
            timestamp = timestamp[:timestamp.index('.')]
            timestamp = timestamp.replace(' ', '_').replace(':', '-')
            self.save_path = 'training_' + timestamp

        os.mkdir(self.save_path)
        # Model
        self.model.to(self.device)

        # Optimizer
        self.optimizer = NoamOpt(config["model"]["d_model"], 1, 2000,
                                 torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        # Loss
        criterion = LabelSmoothing(size=self.vocab_size, padding_idx=self.pad_token, smoothing=self.label_smoothing,
                                   device=self.device)
        criterion.to(self.device)
        self.loss_computer = SimpleLossCompute(criterion)
        tr_losses = []
        ts_losses = []
        tr_aux_losses = []
        ts_aux_losses = []
        best_ts_loss = sys.float_info.max
        dataset = SongIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                               batch_size=self.batch_size, n_workers=self.n_workers)
        tr_loader, ts_loader = dataset.get_loaders()
        # Train
        for self.epoch in range(self.n_epochs):
            # print("Epoch ", epoch, " over ", n_epochs)
            self.model.train()
            tr_loss, tr_aux_loss = self.run_epoch(tr_loader)
            self.model.eval()
            ts_loss, ts_aux_loss = self.run_epoch(ts_loader)
            print("Epoch {}: TR loss: {:.4f}, TS loss: {:.4f}, Aux TR loss: {:.4f}, Aux TS loss: {:.2f}".format(
                self.epoch, tr_loss, ts_loss, tr_aux_loss, ts_aux_loss, end="\n"))
            # Save model if best and erase all others
            if ts_loss < best_ts_loss:
                print("Saving best model")
                best_ts_loss = ts_loss
                torch.save(self.model, os.path.join(self.save_path, self.model_name + '_' + str(self.epoch) + '.pt'))
                for filename in glob.glob(os.path.join(self.save_path, self.model_name+'*')):
                    os.remove(filename)
            # Plot learning curve and save it
            tr_losses.append(tr_loss)
            ts_losses.append(ts_loss)
            tr_aux_losses.append(tr_aux_loss)
            ts_aux_losses.append(ts_aux_loss)
            plot_path = os.path.join(self.save_path, self.plot_name + '_' + str(self.epoch) + '.png')
            self.plot(tr_losses, ts_losses, tr_aux_losses, ts_aux_losses, plot_path=plot_path)
            # Remove old plot
            if self.epoch > 0:
                os.remove(os.path.join(self.save_path, self.plot_name + '_' + str(self.epoch - 1) + '.png'))

    def generate(self, checkpoint_path=None, sampled_dir=None, note_manager=None):
        if self.model is None:
            assert checkpoint_path
            assert sampled_dir
            # Load model
            model = torch.load(checkpoint_path)
        else:
            model = self.model
            sampled_dir = os.path.join(self.save_path, "sampled")
        model.eval()
        # Create final directory
        if not os.path.exists(sampled_dir):
            os.makedirs(sampled_dir)
        else:
            import shutil
            shutil.rmtree(sampled_dir)
            os.makedirs(sampled_dir)
        # Load data
        _, ts_loader = SongIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                                    batch_size=self.batch_size, n_workers=self.n_workers).get_loaders()
        original_song = ts_loader.__iter__().next()
        original_song = np.array(original_song)
        src = np.swapaxes(original_song, 0, 1)
        src = np.swapaxes(src, 1, 2)
        src = torch.LongTensor(src.astype(np.long)).to(self.device)
        out, _, _ = model.forward(src)
        tracks_tokens = torch.max(out, dim=-2).indices
        song = np.array(tracks_tokens.cpu()).swapaxes(0, 1)
        song = song[0, :, :, :]
        final = np.zeros((4, 100, 99), dtype=np.int16)
        final[0] = song[:, :, 0]
        final[1] = song[:, :, 1]
        final[2] = song[:, :, 2]
        final[3] = song[:, :, 3]
        # song = np.swapaxes(song, 0, 1)
        # song = np.swapaxes(song, 0, 3)
        # song = np.reshape(song, (-1, 4))
        # song = np.swapaxes(song, 0, 1)
        final = note_manager.reconstruct_music(final)
        final.write_midi(os.path.join(sampled_dir, "after.mid"))
        # Experiment  # TODO save song in right format to be listened
        # song_one = np.reshape(song[:, :, 0, :], (-1, 4)).swapaxes(0, 1)
        original_song = original_song[0]
        original_song = original_song.reshape(4, -1)
        original_song = note_manager.reconstruct_music(original_song)
        original_song.write_midi(os.path.join(sampled_dir, "before.mid"))
        song_two = np.reshape(np.array(src[:, :, 0, :].cpu()), (-1, 4)).swapaxes(0, 1)
        # created = note_manager.reconstruct_music(song_one)
        # original = note_manager.reconstruct_music(song_two)
        # created.write_midi(os.path.join(sampled_dir, "after.mid"))
        # original.write_midi(os.path.join(sampled_dir, "before.mid"))
        print("Songs generated in " + sampled_dir)


if __name__ == "__main__":

    set_freer_gpu()
    notes = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])

    # shutil.rmtree(config["paths"]["dataset_path"], ignore_errors=True)
    # notes.convert_dataset()

    m = TransformerAutoencoder(**config["model"])

    trainer = Trainer(model=m,
                      pad_token=config["tokens"]["pad_token"],
                      dataset_path=config["paths"]["dataset_path"],
                      **config["train"])
    trainer.train()
    trainer.generate(note_manager=notes)
