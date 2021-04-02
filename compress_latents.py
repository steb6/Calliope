import torch.nn as nn
from config import config, max_bar_length, n_bars
import torch.nn.functional as F
import torch


# class LatentCompressor(nn.Module):
#     def __init__(self, d_model=config["model"]["d_model"]):
#         super(LatentCompressor, self).__init__()
#         self.comp_bar = nn.Linear(d_model*4, d_model)
#         self.comp_bars = nn.Linear(d_model*n_bars, d_model)
#         self.norm = nn.LayerNorm(d_model)
#
#     def forward(self, latents):
#         n_batch, n_track, seq_len, d_model = latents[0].shape  # out: 1 4 200 256
#         out_latents = []
#
#         for latent in latents:
#             latent = latent.transpose(0, 1)  # out: 4 1 200 256
#
#             z_drums = torch.mean(latent[0], dim=1)
#             z_bass = torch.mean(latent[1], dim=1)
#             z_guitar = torch.mean(latent[2], dim=1)
#             z_strings = torch.mean(latent[3], dim=1)  # 1 256
#
#             z = torch.cat([z_drums, z_bass, z_guitar, z_strings], dim=-1)
#             z = self.comp_bar(z)
#             z = self.norm(z)
#             z = F.leaky_relu(z)
#             out_latents.append(z)
#
#         out_latents = torch.stack(out_latents, dim=1)  # 4 1 256
#         out_latents = out_latents.reshape(n_batch, -1)
#         out_latents = self.comp_bars(out_latents)  # 1 256
#
#         return out_latents
#
#
# class LatentDecompressor(nn.Module):
#     def __init__(self, d_model=config["model"]["d_model"]):
#         super(LatentDecompressor, self).__init__()
#         self.decomp_bars = nn.Linear(d_model, d_model*n_bars)
#         self.d_model = d_model
#
#     def forward(self, latent):  # 1 1000
#         n_batch = latent.shape[0]  # 1 256
#
#         latent = self.decomp_bars(latent)  # 1 1024
#         latent = latent.reshape(n_batch, n_bars, self.d_model)
#
#         latent = latent.transpose(0, 1)  # 4 1 256
#         latent = latent.unsqueeze(-2)  # 4 1 1 256
#         out_latents = []
#         for l in latent:
#             out_latents.append(l)
#
#         return out_latents

# layer -> leaky -> (layer -> norm -> leaky)* -> layer

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

    def forward(self, latents):  # TODO multibar
        """
        :param latents: list(n_batch n_bars n_tok d_model)
        :return: n_batch d_model
        """
        out_latents = []
        n_batch, n_track, seq_len, d_model = latents[0].shape  # out: 1 4 200 256

        for latent in latents:
            latent = self.compressor1(latent)  # out: 1 4 200 32
            latent = F.leaky_relu(latent)

            latent = latent.reshape(n_batch, n_track, seq_len*(d_model//8))  # out: 1 4 6400

            latent = self.compressor2(latent)  # out: 1 4 256
            latent = self.norm2(latent)
            latent = F.leaky_relu(latent)

            latent = latent.reshape(n_batch, d_model*4)  # out: 1 1024

            latent = self.compressor3(latent)  # out: 1 256
            latent = self.norm3(latent)
            latent = F.leaky_relu(latent)

            out_latents.append(latent)

        latents = torch.stack(out_latents, dim=1)
        latents = latents.reshape(n_batch, -1)

        latent = self.compressor_simple(latents)
        return latent


# TODO casual norm
class LatentDecompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(LatentDecompressor, self).__init__()
        self.seq_len = max_bar_length
        self.d_model = d_model
        self.decompressor1 = nn.Linear(d_model//8, d_model)
        self.norm1 = nn.LayerNorm(max_bar_length*(d_model//8))
        self.decompressor2 = nn.Linear(d_model, max_bar_length*(d_model//8))
        self.norm2 = nn.LayerNorm(d_model*4)
        self.decompressor3 = nn.Linear(d_model, d_model*4)
        self.norm0 = nn.LayerNorm(d_model)

        self.decompressor_simple = nn.Linear(d_model, d_model*n_bars)

    def forward(self, latent):  # 1 1000
        """
        # TODO add list
        :param latent: n_batch d_model
        :return: list(n_batch n_track n_tok d_model)
        """
        n_batch = latent.shape[0]  # 1 256
        out_latents = []

        latents = self.decompressor_simple(latent)
        latents = F.leaky_relu(latents)

        latents = latents.reshape(n_batch, n_bars, self.d_model)
        latents = latents.transpose(0, 1)  # n_bars n_batch d_model

        for latent in latents:

            latent = self.decompressor3(latent)  # out: 1 1024
            latent = self.norm2(latent)
            latent = F.leaky_relu(latent)

            latent = latent.reshape(n_batch, 4, self.d_model)  # out: 1 4 256

            latent = self.decompressor2(latent)  # out:  # 1 4 6400
            latent = self.norm1(latent)
            latent = F.leaky_relu(latent)

            latent = latent.reshape(n_batch, 4, max_bar_length, self.d_model//8)  # out: 1 4 200 32

            latent = self.decompressor1(latent)  # out: 1 4 200 256
            out_latents.append(latent)

        return out_latents
