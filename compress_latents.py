import torch.nn as nn
import torch


class CompressLatents(nn.Module):
    def __init__(self, d_model=None, seq_len=None, n_latents=None, z_i_dim=None, z_tot_dim=None):
        super(CompressLatents, self).__init__()
        self.sequence_compressor = nn.Linear((seq_len+1)*d_model, z_i_dim)  # +1 because adding sos and eos
        self.track_compressor = nn.Linear(z_i_dim*n_latents, z_tot_dim)
        self.song_compressor = nn.Linear(z_tot_dim*4, z_tot_dim)
        self.act = nn.ReLU()
        self.compressor = nn.Linear(d_model*4, d_model)

    def forward(self, latents):
        n_batch, n_track, n_latents, n_tok, dim = latents.shape  # 1 x 4 x 10 x 301 x 32
        latents = latents.reshape(n_batch, n_track, n_latents, -1)  # 1 x 4 x 10 x 9632
        latents = self.sequence_compressor(latents)  # 1 x 4 x 10 x 1024
        latents = latents.reshape(n_batch, n_track, -1)  # 1 x 4 x 10240
        latents = self.track_compressor(latents)  # 1 x 4 x 1024
        latents = latents.reshape(n_batch, -1)  # 1 x 4096
        latents = self.song_compressor(latents)
        return latents
        # latents = torch.flatten(latents, start_dim=3)
        # latents = self.sequence_compressor(latents)
        # latents = self.act(latents)
        # latents = torch.flatten(latents, start_dim=2)
        # latents = self.track_compressor(latents)
        # latents = self.act(latents)
        # latents = torch.flatten(latents, start_dim=1)
        # latents = self.song_compressor(latents)
        # return latents


class DecompressLatents(nn.Module):
    def __init__(self, d_model=None, seq_len=None, n_latents=None, z_i_dim=None, z_tot_dim=None):
        super(DecompressLatents, self).__init__()
        self.sequence_decompressor = nn.Linear(z_i_dim, (seq_len+1)*d_model)  # +1 because adding sos and eos
        self.track_decompressor = nn.Linear(z_tot_dim, z_i_dim*n_latents)
        self.song_decompressor = nn.Linear(z_tot_dim, z_tot_dim*4)
        self.z_tot_dim = z_tot_dim
        self.z_i_dim = z_i_dim
        self.n_latents = n_latents
        self.seq_len = seq_len
        self.d_model = d_model
        self.act = nn.ReLU()
        self.decompress(d_model, d_model*4)

    def forward(self, latents):  # 1 x 1024
        n_batch, _ = latents.shape
        latents = self.song_decompressor(latents)  # 1 x 4096
        # latents = self.act(latents)
        latents = latents.reshape(n_batch, 4, self.z_tot_dim)  # 1 x 4 x 1024
        latents = self.track_decompressor(latents)  # 1 x 4 x 10240
        # latents = self.act(latents)
        latents = latents.reshape(n_batch, 4, self.n_latents, self.z_i_dim)  # 1 x 4 x 10 x 1024
        latents = self.sequence_decompressor(latents)  # 1 x 4 x 10 x 9632
        latents = latents.reshape(n_batch, 4, self.n_latents, self.seq_len+1, self.d_model)  # 1 x 4 x 10 x 301 x 32
        return latents
        # n_batch, n_tok, dim = latents.shape  # 1 x 3010 x 32
        # latents = self.decompress(latents)  # 1 x 3010 x 128
        # latents = latents.reshape(n_batch, 4, n_tok, self.d_model)  # 1 x 4 x 3010 x 32
        # latents = latents.reshape(n_batch, -1, self.seq_len, self.d_model*4)  # 1 x 4 x 10 x 301 x 128
        # return latents
        #
        # latents = self.song_decompressor(latents)
        # latents = self.act(latents)
        # latents = latents.reshape(*latents.shape[:-1], 4, self.z_tot_dim)
        # latents = self.track_decompressor(latents)
        # latents = self.act(latents)
        # latents = latents.reshape(*latents.shape[:-1], self.n_latents, self.z_i_dim)
        # latents = self.sequence_decompressor(latents)
        # latents = latents.reshape(*latents.shape[:-1], self.seq_len+1, self.d_model)
        # return latents
