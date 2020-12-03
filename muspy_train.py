import matplotlib.pyplot as plt
import os
import torch
from datetime import datetime
from tqdm.auto import tqdm
from my_compressive_transformer import TransformerAutoencoder
from muspy_config import config
from iterate_dataset import SongIterator
from optimizer import NoamOpt
from label_smoother import LabelSmoothing
from loss_computer import SimpleLossCompute
import numpy as np
from create_dataset import NoteRepresentationManager
import shutil


# TO CONNECT: ssh berti@131.114.137.168
# TO VISUALIZE GPUs STATUS: watch -n 0.5 nvidia-smi

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

    def make_masks(self, s):
        tx = s[:, :, :-1]
        ty = s[:, :, 1:]
        sm = s != self.pad_token
        # Create tgt_mask (must be lower diagonal with pad set to false
        mask_subsequent = np.triu(np.ones((tx.shape[-1], tx.shape[-1])), k=1) == 0
        masks_subsequent = np.full((tx.shape[0], tx.shape[1], tx.shape[-1], tx.shape[-1]), True)
        masks_subsequent[:, :] = mask_subsequent  # each track and batch are lower diagonal
        tgt_mask_helper = tx != self.pad_token
        tgt_mask_helper = np.tile(tgt_mask_helper, (1, 1, tx.shape[-1]))
        tgt_mask_helper = np.reshape(tgt_mask_helper, (tx.shape[0], tx.shape[1], tx.shape[-1], tx.shape[-1]))
        tm = masks_subsequent & tgt_mask_helper
        # Adapt dimension
        s = np.swapaxes(s, 1, 2)
        tx = np.swapaxes(tx, 1, 2)
        ty = np.swapaxes(ty, 1, 2)
        sm = np.swapaxes(sm, 1, 2)
        tm = np.swapaxes(tm, 1, 3)
        # Transfer on device
        s = torch.LongTensor(s).to(self.device)  # cant use IntTensor for embedding
        tx = torch.LongTensor(tx).to(self.device)
        ty = torch.LongTensor(ty).to(self.device)
        sm = torch.BoolTensor(sm).to(self.device)
        tm = torch.BoolTensor(tm).to(self.device)
        return s, tx, ty, sm, tm

    def run_epoch(self, loader, memories):
        total_loss = 0
        total_aux_loss = 0
        tokens = 0
        i = 0
        description = ("Train" if self.model.training else "Eval ") + " epoch " + str(self.epoch)
        length = loader.train_len() if self.model.training else loader.test_len()
        progbar = tqdm(total=length, desc=description, leave=True, position=0)
        src, new = loader.get_train_elem() if self.model.training else loader.get_test_elem()
        while src is not None:  # for each batch in the epoch
            src, tgt_x, tgt_y, src_mask, tgt_mask = self.make_masks(src)
            n_tokens = torch.sum(src_mask)
            if n_tokens == 0:  # All tracks are empty, it can happen because first bar usually is empty
                # src, new = loader.get_train_elem() if self.model.training else loader.get_test_elem()
                src, new = loader.get_train_elem() if self.model.training else loader.get_test_elem()
                continue
            # Step
            out, memories, aux_loss = self.model.forward(src, tgt_x, src_mask, tgt_mask, memories=memories)
            loss = self.loss_computer(out, tgt_y, n_tokens)
            if self.optimizer is not None:
                self.optimizer.optimizer.zero_grad()
                (aux_loss + loss).backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_aux_loss += aux_loss.item()
            tokens += n_tokens.sum().item()
            i += 1
            src, new = loader.get_train_elem() if self.model.training else loader.get_test_elem()
            progbar.update(new)
        progbar.close()
        if i == 0:
            i = 1  # TODO why sometimes is 0? check the loader
        return total_loss / i, total_aux_loss / i, memories

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
        self.optimizer = NoamOpt(self.model.src_embed[0].d_model, 1, 2000,
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
        # Init empty memories
        memories = None
        # Train
        for self.epoch in range(self.n_epochs):
            # print("Epoch ", epoch, " over ", n_epochs)
            dataset = SongIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                                   batch_size=self.batch_size, n_workers=self.n_workers)
            self.model.train()
            tr_loss, tr_aux_loss, memories = self.run_epoch(dataset, memories)
            self.model.eval()
            ts_loss, ts_aux_loss, _ = self.run_epoch(dataset, memories)
            print("Epoch {}: TR loss: {:.4f}, TS loss: {:.4f}, Aux TR loss: {:.4f}, Aux TS loss: {:.2f}".format(
                self.epoch, tr_loss, ts_loss, tr_aux_loss, ts_aux_loss, end="\n"))
            # Save model TODO keep only best model
            torch.save(self.model, os.path.join(self.save_path, self.model_name + '_' + str(self.epoch) + '.pt'))
            # Plot learning curve and save it
            tr_losses.append(tr_loss)
            ts_losses.append(ts_loss)
            tr_aux_losses.append(tr_aux_loss)
            ts_aux_losses.append(ts_aux_loss)
            plot_path = os.path.join(self.save_path, self.plot_name + '_' + str(self.epoch) + '.png')
            self.plot(tr_losses, ts_losses, tr_aux_losses, ts_aux_losses, plot_path=plot_path)
            # Remove old files  TODO keep only best model
            if self.epoch > 0:
                os.remove(os.path.join(self.save_path, self.model_name + '_' + str(self.epoch - 1) + '.pt'))
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
        src, _ = SongIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                              batch_size=self.batch_size, n_workers=self.n_workers).get_test_elem()
        src, tgt_x, tgt_y, src_mask, tgt_mask = self.make_masks(src)

        out, _, _ = model.forward(src, tgt_x, src_mask, tgt_mask)
        tracks_tokens = torch.max(out, dim=-2).indices
        song = np.array(tracks_tokens.cpu()).swapaxes(1, 2)

        # Experiment
        created = note_manager.reconstruct_music(song[0])
        original = note_manager.reconstruct_music(np.array(src[0].cpu()).swapaxes(0, 1))
        created.write_midi(os.path.join(sampled_dir, "after.mid"))
        original.write_midi(os.path.join(sampled_dir, "before.mid"))


if __name__ == "__main__":
    note_manager = NoteRepresentationManager(**config["tokens"], **config["data"], **config["paths"])

    shutil.rmtree(config["paths"]["dataset_path"], ignore_errors=True)
    note_manager.convert_dataset()

    m = TransformerAutoencoder(**config["model"])

    trainer = Trainer(model=m,
                      pad_token=config["tokens"]["pad_token"],
                      dataset_path=config["paths"]["dataset_path"],
                      **config["train"])
    trainer.train()
    trainer.generate(note_manager=note_manager)
