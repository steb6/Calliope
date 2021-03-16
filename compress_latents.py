import torch.nn as nn
from config import config, max_bar_length
import torch.nn.functional as F


class LatentCompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(LatentCompressor, self).__init__()
        # self.compressor1 = nn.Linear(d_model*4, d_model)  # TODO prima tracce poi insieme
        # self.norm1 = nn.LayerNorm(d_model)
        # self.compressor2 = nn.Linear(d_model, d_model//8)
        # self.norm2 = nn.LayerNorm(d_model//8)
        # self.compressor3 = nn.Linear(max_bar_length*(d_model//8), 1000)
        # self.norm3 = nn.LayerNorm(1000)
        # self.compressor4 = nn.Linear(1000, 256)
        # 1 4 200 256
        # 1 4 200 32
        # 1 4 6400
        # 1 4 256
        # 1 4 1024
        # 1 4 256
        self.compressor1 = nn.Linear(d_model, d_model//8)
        self.norm1 = nn.LayerNorm(d_model//8)
        self.compressor2 = nn.Linear(max_bar_length*(d_model//8), d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.compressor3 = nn.Linear(d_model*4, d_model)

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

        return latent
        # latent = latent.reshape(n_batch, seq_len, d_model*4)  # 1 200 1024
        #
        # latent = F.dropout(self.compressor1(latent), p=0.1, training=self.training)  # 1 200 256
        # latent = self.norm1(latent)
        #
        # latent = F.dropout(self.compressor2(latent), p=0.1, training=self.training)  # 1 200 32
        # latent = self.norm2(latent)
        # latent = F.leaky_relu(latent, 0.2)
        #
        # latent = latent.reshape(n_batch, seq_len*(d_model//8))  # 1 6400
        # latent = self.compressor3(latent)  # 1 1000
        # latent = F.leaky_relu(latent, 0.2)
        #
        # latent = self.compressor4(latent)
        #
        # return latent
        #
        # x = F.dropout(self.lin1(x), p=self.dropout, training=self.training)
        # x = self.norm1(x)
        # x = F.dropout(self.lin2(x), p=self.dropout, training=self.training)
        # x = self.norm2(x)
        # x = F.leaky_relu(x)
        # x = self.lin3(x)


class LatentDecompressor(nn.Module):
    def __init__(self, d_model=config["model"]["d_model"]):
        super(LatentDecompressor, self).__init__()
        # self.decompressor4 = nn.Linear(256, 1000)
        # self.norm4 = nn.LayerNorm(1000)
        # self.decompressor3 = nn.Linear(1000, max_bar_length*(d_model//8))
        # self.norm3 = nn.LayerNorm(max_bar_length*(d_model//8))
        # self.decompressor2 = nn.Linear(d_model//8, d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.decompressor1 = nn.Linear(d_model, d_model*4)
        self.n_batch = config["train"]["batch_size"]
        self.seq_len = max_bar_length
        self.d_model = d_model
        # 1 4 200 256
        # 1 4 200 32
        # 1 4 6400
        # 1 4 256
        # 1 4 1024
        # 1 4 256
        # 1 256
        self.decompressor1 = nn.Linear(d_model//8, d_model)
        self.norm1 = nn.LayerNorm(d_model//8)
        self.decompressor2 = nn.Linear(d_model, max_bar_length*(d_model//8))
        self.norm2 = nn.LayerNorm(d_model)
        self.decompressor3 = nn.Linear(d_model, d_model*4)
        self.norm0 = nn.LayerNorm(d_model)

    def forward(self, latent):  # 1 1000
        latent = self.decompressor3(latent)  # out: 1 1024
        latent = latent.reshape(self.n_batch, 4, self.d_model)  # out: 1 4 256
        latent = self.norm2(latent)

        latent = self.decompressor2(latent)  # out:  # 1 4 6400
        latent = latent.reshape(self.n_batch, 4, max_bar_length, self.d_model//8)  # out: 1 4 200 32
        latent = self.norm1(latent)
        latent = F.leaky_relu(latent)

        latent = self.decompressor1(latent)  # out: 1 4 200 256
        latent = self.norm0(latent)

        # latent = F.dropout(self.decompressor4(latent), p=0.1, training=self.training)
        # latent = self.norm4(latent)
        #
        # latent = F.dropout(self.decompressor3(latent), p=0.1, training=self.training)  # 1 6400
        # latent = self.norm3(latent)
        #
        # latent = latent.reshape(self.n_batch, self.seq_len, (self.d_model//8))  # 1 200 32
        #
        # latent = F.dropout(self.decompressor2(latent), p=0.1, training=self.training)  # 1 200 256
        # latent = self.norm2(latent)
        # latent = F.leaky_relu(latent, 0.2)
        return latent
