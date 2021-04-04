import torch.nn as nn
from config import config, max_bar_length, n_bars
import torch.nn.functional as F
import torch


class LatentCompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(LatentCompressor, self).__init__()
        self.comp_drums = Compressor()
        self.comp_guitar = Compressor()
        self.comp_bass = Compressor()
        self.comp_strings = Compressor()

        self.compress_bar = nn.Linear(d_model*4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.compress_bars = nn.Linear(d_model*n_bars, d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, latents):  # TODO multibar
        """
        :param latents: list(n_batch n_bars n_tok d_model)
        :return: n_batch d_model
        # """
        out_latents = []
        n_track, n_batch, seq_len, d_model = latents[0].shape  # out: 1 4 200 256

        for latent in latents:
            # latent = latent.transpose(0, 1)  # out: 4 1 200 256

            z_drum = self.comp_drums(latent[0])  # out 1 256
            z_guitar = self.comp_guitar(latent[1])  # out 1 256
            z_bass = self.comp_bass(latent[2])  # out 1 256
            z_strings = self.comp_strings(latent[3])  # out 1 256

            latent = torch.stack([z_drum, z_guitar, z_bass, z_strings], dim=1)  # 1 4 256

            latent = latent.reshape(n_batch, d_model*4)  # out: 1 1024

            latent = self.compress_bar(latent)  # out: 1 256
            latent = self.norm(latent)
            latent = F.leaky_relu(latent)

            out_latents.append(latent)

        latents = torch.stack(out_latents, dim=1)
        latents = latents.reshape(n_batch, -1)  # out: 1 1024

        latent = self.compress_bars(latents)  # out: 1 256
        return latent


class Compressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(Compressor, self).__init__()
        self.compressor1 = nn.Linear(d_model, d_model//8)
        # self.norm1 = nn.LayerNorm(d_model//8)
        self.compressor2 = nn.Linear(max_bar_length*(d_model//8), d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, latent):
        """
        :param latent: n_batch n_tok d_model
        :return: n_batch d_model
        """
        n_batch, seq_len, d_model = latent.shape

        latent = self.compressor1(latent)  # out: 1 200 32
        latent = F.leaky_relu(latent)

        latent = latent.reshape(n_batch, seq_len * (d_model // 8))  # out: 1 6400

        latent = self.compressor2(latent)  # out: 1 256
        latent = self.norm2(latent)
        latent = F.leaky_relu(latent)

        return latent


class Decompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(Decompressor, self).__init__()
        self.decompressor1 = nn.Linear(d_model, max_bar_length*(d_model//8))
        self.norm1 = nn.LayerNorm(max_bar_length*(d_model//8))
        self.decompressor2 = nn.Linear(d_model//8, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, latent):
        """
        :param latent: n_batch d_model
        :return: n_batch n_tok d_model
        """
        n_batch, d_model = latent.shape  # 1 256

        latent = self.decompressor1(latent)  # out: 1 6400
        latent = self.norm1(latent)
        latent = F.leaky_relu(latent)

        latent = latent.reshape(n_batch, max_bar_length, (d_model // 8))  # out: 1 200 32

        latent = self.decompressor2(latent)  # out: 1 200 256
        latent = self.norm2(latent)
        latent = F.leaky_relu(latent)

        return latent


class LatentDecompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(LatentDecompressor, self).__init__()
        self.seq_len = max_bar_length
        self.d_model = d_model

        self.decomp_drums = Decompressor()
        self.decomp_bass = Decompressor()
        self.decomp_guitar = Decompressor()
        self.decomp_strings = Decompressor()

        self.norm = nn.LayerNorm(d_model*4)
        self.decompress_bar = nn.Linear(d_model, d_model*4)
        self.decompressor_bars = nn.Linear(d_model, d_model*n_bars)

    def forward(self, latent):  # 1 1000
        """
        # TODO add list
        :param latent: n_batch d_model
        :return: list(n_batch n_track n_tok d_model)
        """
        n_batch = latent.shape[0]  # 1 256
        out_latents = []

        latents = self.decompressor_bars(latent)  # 1 1024
        latents = F.leaky_relu(latents)

        latents = latents.reshape(n_batch, n_bars, self.d_model)  # 1 4 256
        latents = latents.transpose(0, 1)  # 4 1 256

        for latent in latents:

            latent = self.decompress_bar(latent)  # out: 1 1024
            latent = self.norm(latent)
            latent = F.leaky_relu(latent)

            latent = latent.reshape(n_batch, 4, self.d_model)  # out: 1 4 256

            latent = latent.transpose(0, 1)  # out 4 1 256

            z_drums = self.decomp_drums(latent[0])  # out 1 200 256
            z_guitar = self.decomp_guitar(latent[1])  # out 1 200 256
            z_bass = self.decomp_bass(latent[2])  # out 1 200 256
            z_strings = self.decomp_strings(latent[3])  # out 1 200 256

            latent = torch.stack([z_drums, z_guitar, z_bass, z_strings])  # 1 4 200 256

            out_latents.append(latent)

        return out_latents