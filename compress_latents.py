import torch.nn as nn
from config import config
import torch


class LatentCompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(LatentCompressor, self).__init__()
        self.compressor = nn.Linear(d_model*4, d_model)

    def forward(self, latent):
        n_track, n_batch, seq_len, d_model = latent.shape
        # latent = latent.reshape(n_batch, seq_len, d_model*4)
        # latent = self.compressor(latent)
        return latent


class LatentsCompressor(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 seq_len=config["model"]["seq_len"],
                 n_latents=config["train"]["truncated_bars"],
                 ):
        super(LatentsCompressor, self).__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_latents = n_latents

        self.comp1 = nn.Linear(seq_len*d_model, seq_len*(d_model//4))
        self.comp2 = nn.Linear(seq_len*(d_model//4)*4, seq_len*(d_model//4))
        self.comp3 = nn.Linear(seq_len*(d_model//4)*n_latents, seq_len*d_model)

        self.norm1 = nn.LayerNorm(seq_len*(d_model//4))
        self.norm2 = nn.LayerNorm(seq_len*(d_model//4))

        self.compress_track = nn.Linear(d_model*4, d_model)

        self.compress1 = nn.Linear(seq_len*d_model, seq_len*d_model//4)
        self.compress2 = nn.Linear(n_latents*seq_len*d_model//4, seq_len*d_model)
        self.act = nn.ReLU()

    def forward(self, latents):
        # TODO RESHAPING
        # latents = latents.reshape(*latents.shape[:-2], self.seq_len*self.d_model)
        # latents = self.comp1(latents)
        # latents = self.norm1(latents)
        # latents = latents.reshape(*latents.shape[:-2], self.seq_len*(self.d_model//4)*4)
        # latents = self.comp2(latents)
        # latents = self.norm2(latents)
        # latents = latents.reshape(*latents.shape[:-2], self.seq_len*(self.d_model//4)*self.n_latents)
        # latents = self.comp3(latents)
        # JUST MEAN
        # latents = torch.mean(latents, dim=-2)
        # latents = torch.mean(latents, dim=-2)
        # latents = torch.mean(latents, dim=-2, keepdim=True)
        # MIX INSTRUMENTS
        # TODO WORKING SIMPLE COMPRESSION
        n_batch, n_latents, n_track, seq_len, d_model = latents.shape
        latents = latents.reshape(n_batch, n_latents, seq_len, d_model*4)
        latents = self.compress_track(latents)
        # latents = torch.mean(latents, dim=1, keepdim=True)  # compress along time
        # TODO NOW COMPRESS FOR REAL
        latents = self.act(latents)
        latents = latents.reshape(n_batch, n_latents, seq_len*d_model)
        latents = self.compress1(latents)
        # TODO FINAL COMPRESSION
        latents = self.act(latents)
        latents = latents.reshape(n_batch, n_latents*seq_len*d_model//4)
        latents = self.compress2(latents)
        return latents


class LatentsDecompressor(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 seq_len=config["model"]["seq_len"],
                 n_latents=config["train"]["truncated_bars"],
                 ):
        super(LatentsDecompressor, self).__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.seq_len = seq_len

        self.decomp1 = nn.Linear(seq_len*d_model, seq_len*d_model)

        self.norm1 = nn.LayerNorm(seq_len*d_model)

        self.decompress1 = nn.Linear(seq_len*d_model//4, seq_len*d_model)
        self.decompress2 = nn.Linear(seq_len*d_model, n_latents*seq_len*d_model//4)
        self.act = nn.ReLU()

    def forward(self, latents):
        n_batch, z_dim = latents.shape
        # TODO DECOMPRESS BARS
        latents = self.decompress2(latents)
        latents = latents.reshape(n_batch, self.n_latents, self.seq_len*self.d_model//4)
        latents = self.act(latents)
        # TODO DECOMPRESS BAR
        latents = self.decompress1(latents)
        latents = latents.reshape(n_batch, self.n_latents, self.seq_len, self.d_model)
        # n_batch = latents.shape[0]
        # latents = self.decomp1(latents)
        # latents = latents.reshape(n_batch, self.seq_len, self.d_model)
        return latents  # 3 x 4 x 12 x 5 x 32
