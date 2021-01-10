import torch.nn as nn
from config import config
import torch


class LatentsCompressor(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 seq_len=config["model"]["seq_len"],
                 n_latents=config["model"]["total_seq_len"] // config["model"]["seq_len"],
                 z_i_dim=config["model"]["z_i_dim"],
                 z_tot_dim=config["model"]["z_tot_dim"]
                 ):
        super(LatentsCompressor, self).__init__()

        # EXPERIMENTAL
        self.compress_token = nn.Linear(d_model, d_model // 4)
        self.compress_sequence = nn.Linear(seq_len * d_model // 4, seq_len * d_model // 16)
        self.compress_instrument = nn.Linear(n_latents * seq_len * d_model // 16, n_latents * seq_len * d_model // 32)
        # TODO careful in the following passage!
        # self.compress_tracks = nn.Linear(4 * n_latents * seq_len * d_model // 32, n_latents * seq_len * d_model // 32)

        # NEW EXPERIMENTAL
        self.act = torch.nn.LeakyReLU()
        self.final_act = torch.nn.Tanh()
        self.compress_latents = nn.Linear(n_latents*d_model, n_latents*d_model)
        self.compress_tracks = nn.Linear(n_latents*d_model*4, n_latents*d_model*4)
        self.norm1 = torch.nn.LayerNorm(n_latents*d_model)
        self.norm2 = torch.nn.LayerNorm(n_latents * d_model * 4)

    def forward(self, latents):
        n_batch, n_track, n_latents, n_tok, dim = latents.shape  # 1 x 4 x 6 x 100 x 32
        # latents = self.compress_token(latents)  # 1 x 4 x 6 x 100 x 8
        # latents = self.act(latents)
        # latents = latents.reshape(n_batch, n_track, n_latents, -1)  # 1 x 4 x 6 x 800
        # latents = self.compress_sequence(latents)  # 1 x 4 x 6 x 200
        # latents = self.act(latents)
        # latents = latents.reshape(n_batch, n_track, -1)  # 1 x 4 x 1200
        # latents = self.compress_instrument(latents)  # 1 x 4 x 600  # TODO till here ok
        # latents = latents.reshape(n_batch, -1)  # 1 x 2400
        # latents = self.compress_tracks(latents)

        latents = torch.mean(latents, dim=-2)
        latents = latents.reshape(n_batch, n_track, -1)  # 1 x 4 x 384
        latents = self.compress_latents(latents)
        latents = self.norm1(latents)
        latents = latents.reshape(n_batch, -1)  # 1 x 1536
        latents = self.compress_tracks(latents)
        latents = self.norm2(latents)
        return latents


class LatentsDecompressor(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 seq_len=config["model"]["seq_len"],
                 n_latents=config["model"]["total_seq_len"] // config["model"]["seq_len"],
                 z_i_dim=config["model"]["z_i_dim"],
                 z_tot_dim=config["model"]["z_tot_dim"]
                 ):
        super(LatentsDecompressor, self).__init__()
        self.z_tot_dim = z_tot_dim
        self.z_i_dim = z_i_dim
        self.n_latents = n_latents
        self.seq_len = seq_len
        self.d_model = d_model
        # EXPERIMENTAL
        # self.decompress_tracks = nn.Linear(n_latents * seq_len * d_model // 32, 4 * n_latents * seq_len * d_model // 32)
        self.decompress_instrument = nn.Linear(n_latents * seq_len * d_model // 32, n_latents * seq_len * d_model // 16)
        self.decompress_sequence = nn.Linear(seq_len * d_model // 16, seq_len * d_model // 4)
        self.decompress_token = nn.Linear(d_model // 4, d_model)
        # NEW EXPERIMENTAL
        self.act = torch.nn.LeakyReLU()
        self.final_act = torch.nn.Tanh()
        self.decompress_latents = nn.Linear(n_latents*d_model, n_latents*d_model)
        self.decompress_tracks = nn.Linear(n_latents*d_model*4, n_latents*d_model*4)
        self.norm1 = nn.LayerNorm(n_latents*d_model*4)
        self.norm2 = nn.LayerNorm(d_model*n_latents)

    def forward(self, latents):  # 1 x 1024 -> 1 x 4 x 10 x 301 x 32
        n_batch = latents.shape[0]
        # latents = self.decompress_tracks(latents)
        # latents = latents.reshape(n_batch, 4, -1)
        # latents = self.decompress_instrument(latents)
        # latents = self.act(latents)
        # latents = latents.reshape(n_batch, 4, self.n_latents, self.seq_len * self.d_model // 16)
        # latents = self.decompress_sequence(latents)
        # latents = self.act(latents)
        # latents = latents.reshape(n_batch, 4, self.n_latents, self.seq_len, -1)
        # latents = self.decompress_token(latents)

        latents = self.decompress_tracks(latents)
        latents = self.norm1(latents)
        latents = latents.reshape(n_batch, 4, -1)
        latents = self.decompress_latents(latents)
        latents = self.norm2(latents)
        latents = latents.reshape(n_batch, 4, self.n_latents, self.d_model)
        return latents
