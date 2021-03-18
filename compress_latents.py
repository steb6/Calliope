import torch.nn as nn
from config import config, max_bar_length
import torch.nn.functional as F


class LatentCompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(LatentCompressor, self).__init__()
        self.compressor1 = nn.Linear(d_model, d_model//8)
        self.norm1 = nn.LayerNorm(d_model//8)
        self.compressor2 = nn.Linear(max_bar_length*(d_model//8), d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.compressor3 = nn.Linear(d_model*4, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, latent):
        n_batch, n_track, seq_len, d_model = latent.shape  # out: 4 1 200 256

        latent = F.dropout(self.compressor1(latent), p=0.1, training=self.training)  # out: 1 4 200 32
        latent = self.norm1(latent)

        latent = latent.reshape(n_batch, n_track, seq_len*(d_model//8))  # out: 1 4 6400

        latent = self.compressor2(latent)  # out: 1 4 256
        latent = self.norm2(latent)
        latent = F.leaky_relu(latent)

        latent = latent.reshape(n_batch, d_model*4)  # out: 1 1024

        latent = self.compressor3(latent)  # out: 1 256
        # latent = self.norm3(latent)

        return latent


class LatentDecompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(LatentDecompressor, self).__init__()
        self.seq_len = max_bar_length
        self.d_model = d_model
        self.decompressor1 = nn.Linear(d_model//8, d_model)
        self.norm1 = nn.LayerNorm(d_model//8)
        self.decompressor2 = nn.Linear(d_model, max_bar_length*(d_model//8))
        self.norm2 = nn.LayerNorm(d_model)
        self.decompressor3 = nn.Linear(d_model, d_model*4)
        self.norm0 = nn.LayerNorm(d_model)

    def forward(self, latent):  # 1 1000
        n_batch = latent.shape[0]

        latent = self.decompressor3(latent)  # out: 1 1024
        latent = latent.reshape(n_batch, 4, self.d_model)  # out: 1 4 256
        latent = self.norm2(latent)

        latent = self.decompressor2(latent)  # out:  # 1 4 6400
        latent = latent.reshape(n_batch, 4, max_bar_length, self.d_model//8)  # out: 1 4 200 32
        latent = self.norm1(latent)
        latent = F.leaky_relu(latent)

        latent = self.decompressor1(latent)  # out: 1 4 200 256
        latent = self.norm0(latent)

        return latent
