import torch.nn as nn
from config import config, max_bar_length, n_bars
import torch.nn.functional as F
import torch


class LatentCompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(LatentCompressor, self).__init__()
        self.compressor1 = nn.Linear(d_model, d_model//8)
        self.norm1 = nn.LayerNorm(d_model//8)
        self.compressor2 = nn.Linear(max_bar_length*(d_model//8), d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.compressor3 = nn.Linear(d_model*4, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.compressor_simple = nn.Linear(d_model*n_bars, d_model)

    def forward(self, latents):
        n_batch, n_track, seq_len, d_model = latents[0].shape  # out: 1 4 200 256
        out_latents = []

        for latent in latents:
            latent = F.dropout(self.compressor1(latent), p=0.1, training=self.training)  # out: 1 4 200 32
            latent = self.norm1(latent)

            latent = latent.reshape(n_batch, n_track, seq_len*(d_model//8))  # out: 1 4 6400

            latent = F.dropout(self.compressor2(latent), p=0.1, training=self.training)  # out: 1 4 256
            latent = self.norm2(latent)
            latent = F.leaky_relu(latent)

            latent = latent.reshape(n_batch, d_model*4)  # out: 1 1024

            latent = F.dropout(self.compressor3(latent), p=0.1, training=self.training)  # out: 1 256
            latent = self.norm3(latent)
            latent = F.leaky_relu(latent)
            out_latents.append(latent)

        out_latents = torch.stack(out_latents, dim=1)
        out_latents = out_latents.reshape(n_batch, -1)
        out_latents = F.dropout(self.compressor_simple(out_latents), p=0.1, training=self.training)

        return out_latents


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

        self.decompress_simple = nn.Linear(d_model, d_model*n_bars)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, latent):  # 1 1000
        n_batch = latent.shape[0]
        out_latents = []

        latents = F.dropout(self.decompress_simple(latent), p=0.1, training=self.training)
        latents = latents.reshape(n_batch, n_bars, self.d_model)
        latents = self.norm3(latents)
        latents = latents.transpose(0, 1)

        for latent in latents:
            latent = F.dropout(self.decompressor3(latent), p=0.1, training=self.training)  # out: 1 1024
            latent = latent.reshape(n_batch, 4, self.d_model)  # out: 1 4 256
            latent = self.norm2(latent)
            latent = F.leaky_relu(latent)

            latent = F.dropout(self.decompressor2(latent), p=0.1, training=self.training)  # out:  # 1 4 6400
            latent = latent.reshape(n_batch, 4, max_bar_length, self.d_model//8)  # out: 1 4 200 32
            latent = self.norm1(latent)
            latent = F.leaky_relu(latent)

            latent = F.dropout(self.decompressor1(latent), p=0.1, training=self.training)  # out: 1 4 200 256
            latent = self.norm0(latent)
            out_latents.append(latent)

        return out_latents
