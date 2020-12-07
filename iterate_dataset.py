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
            drop_last=False  # if dataset length is not divisible by batch_size, drop last batch
        )

        ts_loader = torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=ts_sampler,
            num_workers=self.n_workers,
            drop_last=False
        )
        return tr_loader, ts_loader


# class SongIterator:
#     def __init__(self, dataset_path=None, test_size=None, batch_size=None, n_workers=None):
#         self.dataset_path = dataset_path
#         self.test_size = test_size
#         self.batch_size = batch_size
#         self.n_workers = n_workers
#         self.n_songs = len([name for name in os.listdir(dataset_path)])
#         songs = list(range(self.n_songs))
#         random.shuffle(songs)
#         self.offset = int(self.n_songs*test_size) if int(self.n_songs*test_size) > batch_size else batch_size
#         if self.n_songs-self.offset < batch_size:
#             raise Exception("Not enough song provided w.r.t the batch size")
#         self.train_songs = songs[self.offset:]
#         self.test_songs = songs[:self.offset]
#         with open(os.path.join(self.dataset_path, str(0) + '.pickle'), 'rb') as f:
#             song = pickle.load(f)
#         self.n_tracks = song.shape[0]
#         self.max_track_length = song.shape[1]
#         self.max_bars = 100
#         self.max_bar_length = 100
#         self.bar_token = 1
#
#     def train_len(self):
#         return self.n_songs - self.offset
#
#     def test_len(self):
#         return self.offset
#
#     def get_elem(self, train):
#         if train:
#             songs = self.train_songs
#         else:
#             songs = self.test_songs
#         empties = 0
#         # mb = np.zeros((self.batch_size, self.n_tracks, self.max_track_length), dtype=np.int16)
#         mb = np.zeros((self.batch_size, self.n_tracks, self.max_bars, self.max_bar_length), dtype=np.int16)
#         for b in range(self.batch_size):
#             if len(songs) == 0:
#                 empties += 1
#                 continue
#             index = songs.pop()
#             with open(os.path.join(self.dataset_path, str(index) + '.pickle'), 'rb') as f:
#                 song = pickle.load(f)
#             mb[b] = song
#
#         if empties == self.batch_size:
#             return None, 1
#         return mb, self.batch_size


# class DatasetIterator(torch.utils.data.Dataset):
#     def __init__(self, dataset_path, test_size, batch_size=3, n_workers=1):
#         self.dataset_path = dataset_path
#         _, _, songs = next(os.walk(self.dataset_path))
#         self.songs = [x.split(".")[0] for x in songs]
#         random.shuffle(self.songs)
#         ts_length = int(len(songs) * test_size)
#         self.ts_set = self.songs[:ts_length]
#         self.tr_set = self.songs[ts_length:]
#         self.batch_size = batch_size
#         self.n_workers = n_workers
#
#     def __getitem__(self, idx):
#         # with open(os.path.join(self.dataset_path, idx+str('.pickle')), 'rb') as file:
#             # bar = pickle.load(file)
#         return bar
#
#     def __len__(self, train=None):
#         if train:
#             return len(self.tr_set)
#         else:
#             return len(self.ts_set)
#
#     def get_loaders(self):
#         tr_sampler = SubsetRandomSampler(self.tr_set)  # TODO random sampling does not ruins flow of a song?
#         ts_sampler = SubsetRandomSampler(self.ts_set)
#
#         tr_loader = torch.utils.data.DataLoader(
#             self,
#             batch_size=self.batch_size,
#             sampler=None,
#             num_workers=self.n_workers,
#             drop_last=False  # if dataset length is not divisible by batch_size, drop last batch
#         )
#
#         ts_loader = torch.utils.data.DataLoader(
#             self,
#             batch_size=self.batch_size,
#             sampler=None,
#             num_workers=self.n_workers,
#             drop_last=False
#         )
#         return tr_loader, ts_loader
#
#
# if __name__ == "__main__":
#     dataset = DatasetIterator("dataset", 0.2)
#     print("Done")

    # def __getitem__(self, idx):
    #     filename = os.path.join(self.dataset_path, str(self.current_song)+'-'+str(self.current_id)+'.pickle')
    #     if not os.path.exists(filename):
    #         if len(self.remained) > 0:
    #             self.current_song = self.remained.pop()
    #             self.current_id = 0
    #         else:
    #             return None  # songs over
    #     with open(os.path.join(self.dataset_path, str(self.current_song)+'-'+str(self.current_id)+'.pickle'), 'rb') as f:
    #         self.current_id += 1
    #         return pickle.load(f)
    #
    #     # with open(os.path.join(self.dataset_path, idx+str('.pickle')), 'rb') as file:
    #     #    bar = pickle.load(file)
    #     # if self.current is None:
    #
    #     # bar = self.current[0]
    #     # self.current = self.current[1:]
    #     # if len(self.current) == 0:
    #     #     self.current = None
    #     # return bar


# class DatasetIterator:  # iterates over bar
#     def __init__(self, dataset_path=None, test_size=None, batch_size=None, n_workers=None, n_songs=None, max_bars=None):
#         self.dataset_path = dataset_path
#         self.test_size = test_size
#         self.batch_size = batch_size
#         self.n_workers = n_workers
#         self.n_songs = n_songs
#         songs = list(range(n_songs))
#         random.shuffle(songs)
#         self.offset = int(n_songs*test_size) if int(n_songs*test_size) > 1 else 1
#         if n_songs-self.offset < batch_size:
#             raise Exception("Number of train song provided must be at least batch size!")
#         if self.offset < batch_size:
#             raise Exception("Number of test song provided must be at least batch size!")
#         self.test_remained = songs[:self.offset]
#         self.test_actual = []
#         self.train_remained = songs[self.offset:]
#         self.train_actual = []
#
#         while len(self.train_actual) < self.batch_size:  # it enters here just the first time
#             index = self.train_remained.pop()
#             with open(os.path.join(self.dataset_path, str(index)+'.pickle'), 'rb') as f:
#                 song = pickle.load(f)
#             self.train_actual.append(song)
#
#         while len(self.test_actual) < self.batch_size:  # it enters here just the first time
#             index = self.test_remained.pop()
#             with open(os.path.join(self.dataset_path, str(index)+'.pickle'), 'rb') as f:
#                 song = pickle.load(f)
#             self.test_actual.append(song)
#
#         self.bar_length = self.train_actual[0].shape[-1]
#         self.n_tracks = self.train_actual[0].shape[0]
#         self.max_bars = max_bars
#
#     def train_len(self):
#         return self.n_songs - self.offset
#
#     def test_len(self):
#         return self.offset
#
#     def get_train_elem(self):
#         empties = 0
#         song_over = 0
#         mb = np.zeros((self.batch_size, self.n_tracks, self.bar_length), dtype=np.int16)
#         for i in range(self.batch_size):
#             if self.train_actual[i] is None:  # song is over and there is no other to load, just continue
#                 empties += 1
#                 continue
#             if np.size(self.train_actual[i], 1) == 0:  # if song is over, load another one
#                 song_over += 1
#                 if len(self.train_remained) > 0:  # if there is a song to load
#                     index = self.train_remained.pop()
#                     with open(os.path.join(self.dataset_path, str(index)+'.pickle'), 'rb') as f:
#                         song = pickle.load(f)
#                     self.train_actual[i] = song
#                 else:  # if no more songs left, just continue (pad is already present)
#                     self.train_actual[i] = None
#                     empties += 1
#                     continue
#             mb[i] = self.train_actual[i][:, 0, :]
#             self.train_actual[i] = self.train_actual[i][:, 1:, :]
#         if empties == self.batch_size:  # we do continue 3 times, so the elem is made only by padding
#             return None, 1
#         return mb, song_over
#
#     def get_test_elem(self):
#         empties = 0
#         song_over = 0
#         mb = np.zeros((self.batch_size, self.n_tracks, self.bar_length), dtype=np.int16)
#         for i in range(self.batch_size):
#             if self.test_actual[i] is None:  # song is over and there is no other to load, just continue
#                 empties += 1
#                 continue
#             if np.size(self.test_actual[i], 1) == 0:  # if song is over, load another one
#                 song_over += 1
#                 if len(self.test_remained) > 0:  # if there is a song to load
#                     index = self.test_remained.pop()
#                     with open(os.path.join(self.dataset_path, str(index)+'.pickle'), 'rb') as f:
#                         song = pickle.load(f)
#                     self.test_actual[i] = song
#                 else:  # if no more songs left, just continue (pad is already present)
#                     self.test_actual[i] = None
#                     empties += 1
#                     continue
#             mb[i] = self.test_actual[i][:, 0, :]
#             self.test_actual[i] = self.test_actual[i][:, 1:, :]
#         if empties == self.batch_size:  # we do continue 3 times, so the elem is made only by padding
#             return None, 1
#         return mb, song_over
#
#
