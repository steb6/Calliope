import torch.utils.data
import os
import random
import pickle
from torch.utils.data import SubsetRandomSampler


class SongIterator(torch.utils.data.Dataset):
    def __init__(self, dataset_path, test_size, batch_size=3, n_workers=1):
        self.dataset_path = dataset_path
        _, _, songs = next(os.walk(self.dataset_path))
        self.songs = [x.split(".")[0] for x in songs]
        random.shuffle(self.songs)
        ts_length = int(len(songs) * test_size)
        self.ts_set = self.songs[:ts_length]
        self.tr_set = self.songs[ts_length:]
        self.batch_size = batch_size
        if len(self.tr_set) < batch_size:
            raise Exception("Training set is too little w.r.t. the batch size")
        if len(self.ts_set) < batch_size:
            raise Exception("Testing set is too little w.r.t. the batch size")
        self.n_workers = n_workers

    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_path, idx+str('.pickle')), 'rb') as file:
            song = pickle.load(file)
        return song

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
