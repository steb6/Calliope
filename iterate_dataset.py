import torch.utils.data
import os
import random
import pickle
from torch.utils.data import SubsetRandomSampler
from config import config
import numpy as np
import pickle


class SongIterator(torch.utils.data.Dataset):
    def __init__(self, dataset_path, batch_size=3, n_workers=1):
        self.dataset_path = dataset_path
        print(dataset_path)
        _, _, song_set = next(os.walk(self.dataset_path))
        self.song_set = song_set
        random.shuffle(self.song_set)

        self.batch_size = batch_size

        if len(self.song_set) < batch_size:
            raise Exception(self.dataset_path, "is too little w.r.t. the batch size")
        self.n_workers = n_workers

    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_path, idx), 'rb') as file:  # +str('.pickle')
            src = pickle.load(file)
        sos = np.full(src.shape[:-1]+(1,), config["tokens"]["sos"], dtype=src.dtype)
        src = np.append(sos, src, axis=-1)
        # put eos
        for instrument in src:
            for bar in instrument:
                idx = np.where(bar == config["tokens"]["pad"])
                bar[idx[0][0]] = config["tokens"]["eos"]
        trg = src[..., :-1]
        src = src[..., 1:]
        return src, trg

    def __len__(self, train=None):
        return len(self.song_set)

    def get_loader(self):
        sampler = SubsetRandomSampler(self.song_set)

        loader = torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.n_workers,
            drop_last=True  # if dataset length is not divisible by batch_size, drop last batch
        )

        return loader
