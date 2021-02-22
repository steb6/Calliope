import torch.utils.data
import os
import random
import pickle
from torch.utils.data import SubsetRandomSampler
from config import config
import numpy as np


class SongIterator(torch.utils.data.Dataset):
    def __init__(self, dataset_path, test_size, max_len=3000, batch_size=3, n_workers=1):
        self.dataset_path = dataset_path
        _, _, songs = next(os.walk(self.dataset_path))
        self.songs = [x.split(".")[0] for x in songs]
        random.shuffle(self.songs)
        ts_length = int(len(songs) * test_size)
        self.ts_set = self.songs[:ts_length]
        self.tr_set = self.songs[ts_length:]
        self.batch_size = batch_size
        self.max_len = max_len
        if len(self.tr_set) < batch_size:
            raise Exception("Training set is too little w.r.t. the batch size")
        if len(self.ts_set) < batch_size:
            raise Exception("Testing set is too little w.r.t. the batch size")
        self.n_workers = n_workers

    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_path, idx+str('.pickle')), 'rb') as file:
            src = pickle.load(file)
        # src = src[:, :(src.shape[1]-src.shape[1] % config["train"]["truncated_bars"]), :]
        # src = src.reshape(4, -1, config["train"]["truncated_bars"], config["model"]["seq_len"])
        sos = np.full(src.shape[:-1]+(1,), config["tokens"]["sos"], dtype=src.dtype)
        src = np.append(sos, src, axis=-1)
        for instrument in src:
            for bar in instrument:
                idx = np.where(bar == config["tokens"]["pad"])
                bar[idx[0][0]] = config["tokens"]["eos"]
        src_mask = src != config["tokens"]["pad"]
        trg = src[..., :-1]
        trg_y = src[..., 1:]
        trg_mask = np.full(trg.shape+(trg.shape[-1],), True)
        for i, instrument in enumerate(trg):
            for b, bar in enumerate(instrument):
                line_mask = bar != config["tokens"]["pad"]
                pad_mask = np.matmul(line_mask[:, np.newaxis], line_mask[np.newaxis, :])
                subsequent_mask = np.expand_dims(np.tril(np.ones((trg.shape[-1], trg.shape[-1]))), (0, 1))
                subsequent_mask = subsequent_mask.astype(np.bool)
                trg_mask[i][b] = pad_mask & subsequent_mask
        src = src[..., 1:]
        src_mask = src_mask[..., 1:]
        return src, trg, src_mask, trg_mask, trg_y

    def __len__(self, train=None):
        if train:
            return len(self.tr_set)
        else:
            return len(self.ts_set)

    def get_loaders(self):
        tr_sampler = SubsetRandomSampler(self.tr_set)  # TODO random sampling does not ruins flow of a song?
        ts_sampler = SubsetRandomSampler(self.ts_set)

        tr_loader = torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=tr_sampler,
            num_workers=self.n_workers,
            drop_last=True  # if dataset length is not divisible by batch_size, drop last batch
        )

        ts_loader = torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=ts_sampler,
            num_workers=self.n_workers,
            drop_last=True
        )
        return tr_loader, ts_loader


        # n_batches, seq_len = seq.shape
        # pad = torch.empty((n_batches, 1), dtype=torch.int64).fill_(config["tokens"]["pad"]
        #                                                            ).to(config["train"]["device"])
        # seq = torch.cat((seq, pad), dim=-1)
        # for b, s in enumerate(seq):  # add eos token
        #     idx = torch.nonzero(s == config["tokens"]["pad"])
        #     if s.shape[0] == idx.shape[0]:
        #         continue  # all pad, do nothing
        #     seq[b][idx[0]] = config["tokens"]["eos"]