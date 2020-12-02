import matplotlib.pyplot as plt
import os
import torch
from datetime import datetime
from tqdm.auto import tqdm
from my_compressive_transformer import TransformerAutoencoder
from muspy_config import config
from iterate_dataset import DatasetIterator
from optimizer import NoamOpt
from label_smoother import LabelSmoothing
from loss_computer import SimpleLossCompute
import numpy as np
from create_dataset import NoteRepresentationManager
import shutil


# TO CONNECT: ssh berti@131.114.137.168


class Trainer:
    def __init__(self, save_path=None, pad_token=512, device="cpu", dataset_path="dataset", test_size=0.1,
                 batch_size=3, n_workers=1, vocab_size=513, n_epochs=140, model_name="checkpoint", plot_name="plot",
                 model=None, dataset=None):
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
        self.dataset = dataset

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
        criterion = LabelSmoothing(size=self.vocab_size, padding_idx=self.pad_token, smoothing=0.1)
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
            dataset = DatasetIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                                      batch_size=self.batch_size, n_workers=self.n_workers, n_songs=config.early_stop)
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

    def generate(self, checkpoint_path=None, sampled_dir=None):
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
        dataset = DatasetIterator(dataset_path=self.dataset_path, test_size=self.test_size,
                                  batch_size=self.batch_size, n_workers=self.n_workers, n_songs=config.early_stop)
        for i in range(5):  # skip first 5 bar, that could be empty, to better test the model
            _, _ = dataset.get_test_elem()
        src, _ = dataset.get_test_elem()
        src, tgt_x, tgt_y, src_mask, tgt_mask = self.make_masks(src)

        out, _, _ = model.forward(src, tgt_x, src_mask, tgt_mask)
        tracks_tokens = torch.max(out, dim=-2).indices
        song = np.array(tracks_tokens.cpu()).swapaxes(1, 2)

        # Experiment
        def append_song(m1, m2):
            for i in range(4):
                for note in m2.tracks[i].notes:
                    m1.tracks[i].notes.append(note)
            return m1

        created1 = self.dataset.reconstruct_music(np.expand_dims(song[0], 1), time_offset=0)
        created2 = self.dataset.reconstruct_music(np.expand_dims(song[1], 1), time_offset=1)
        created3 = self.dataset.reconstruct_music(np.expand_dims(song[2], 1), time_offset=2)
        original1 = self.dataset.reconstruct_music(np.expand_dims(np.array(src[0].cpu()).swapaxes(0, 1), 1),
                                                   time_offset=0)
        original2 = self.dataset.reconstruct_music(np.expand_dims(np.array(src[1].cpu()).swapaxes(0, 1), 1),
                                                   time_offset=1)
        original3 = self.dataset.reconstruct_music(np.expand_dims(np.array(src[2].cpu()).swapaxes(0, 1), 1),
                                                   time_offset=2)
        append_song(append_song(original1, original2), original3).write_midi(os.path.join(sampled_dir, "before.mid"))
        append_song(append_song(created1, created2), created3).write_midi(os.path.join(sampled_dir, "after.mid"))


if __name__ == "__main__":
    data = NoteRepresentationManager(resolution=config.resolution,
                                     tempo=config.tempo,
                                     pad_token=config.pad_token,
                                     sos_token=config.sos_token,
                                     bar_token=config.bar_token,
                                     eos_token=config.eos_token,
                                     num_values=config.num_values,
                                     time_first_token=config.time_first_token,
                                     pitch_first_token=config.pitch_first_token,
                                     duration_first_token=config.duration_first_token,
                                     velocity_first_token=config.velocity_first_token,
                                     use_velocity=config.use_velocity,
                                     log_file=config.dataset_converter_log_file,
                                     reconstruct_programs=config.reconstruct_programs,
                                     max_bar_length=config.max_bar_length
                                     )
    shutil.rmtree(config.dataset_path, ignore_errors=True)
    data.convert_dataset(raw_midi=config.raw_midi_path, dataset_path=config.dataset_path, early_stop=config.early_stop)

    m = TransformerAutoencoder(d_model=config.d_model,
                               n_tracks=config.n_tracks,
                               heads=config.n_heads,
                               d_ff=config.d_ff,
                               dropout=config.dropout,
                               layers=config.layers,
                               vocab_size=config.vocab_size,
                               seq_len=config.seq_len,
                               mem_len=config.mem_len,
                               cmem_len=config.cmem_len,
                               cmem_ratio=config.cmem_ratio)

    trainer = Trainer(model=m,
                      pad_token=config.pad_token,
                      device=config.device,
                      dataset_path=config.dataset_path,
                      test_size=config.test_size,
                      batch_size=config.batch_size,
                      n_workers=config.n_workers,
                      vocab_size=config.vocab_size,
                      n_epochs=config.n_epochs,
                      model_name=config.model_name,
                      plot_name=config.plot_name,
                      dataset=data)
    trainer.train()
    trainer.generate()
