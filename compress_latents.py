import torch.nn as nn
from config import config
import torch


class LatentsCompressor(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 seq_len=config["model"]["seq_len"],
                 n_latents=config["model"]["total_seq_len"] // config["model"]["seq_len"],
                 z_i_dim=config["model"]["z_i_dim"],
                 z_tot_dim=config["model"]["z_tot_dim"],
                 token_reduction=4,
                 sequence_reduction=4,
                 latent_reduction=4,
                 instrument_reduction=4
                 ):
        super(LatentsCompressor, self).__init__()

        # dim_1 = d_model // token_reduction
        # dim_2 = (dim_1 * seq_len) // sequence_reduction
        # dim_3 = (dim_2*n_latents) // latent_reduction
        # dim_4 = (dim_3*4) // instrument_reduction
        # self.compress_token = nn.Linear(d_model, dim_1)  # 32 / 4 = 8
        # self.norm1 = nn.LayerNorm(dim_1)
        # self.compress_sequence = nn.Linear(dim_1*seq_len, dim_2)
        # self.norm2 = nn.LayerNorm(dim_2)
        # self.compress_track = nn.Linear(dim_2*n_latents, dim_3)
        # self.norm3 = nn.LayerNorm(dim_3)
        # self.compress_instruments = nn.Linear(dim_3*4, dim_4)
        # self.norm4 = nn.LayerNorm(dim_4)
        self.comp1 = nn.Linear(n_latents*d_model, n_latents*d_model)
        self.norm1 = nn.LayerNorm(n_latents*d_model)
        self.comp2 = nn.Linear(n_latents*d_model*4, n_latents*d_model*4)
        self.norm2 = nn.LayerNorm(n_latents*d_model*4)

    def forward(self, latents):
        # latents = self.compress_token(latents)
        # latents = self.norm1(latents)
        # latents = torch.flatten(latents, start_dim=-2)
        # latents = self.compress_sequence(latents)
        # latents = self.norm2(latents)
        # latents = torch.flatten(latents, start_dim=-2)
        # latents = self.compress_track(latents)
        # latents = self.norm3(latents)
        # latents = torch.flatten(latents, start_dim=-2)
        # latents = self.compress_instruments(latents)
        # latents = self.norm4(latents)

        latents = torch.mean(latents, dim=-2)
        # latents = torch.flatten(latents, start_dim=-2)
        # latents = self.comp1(latents)
        # latents = self.norm1(latents)
        # latents = torch.flatten(latents, start_dim=-2)
        # latents = self.comp2(latents)
        # latents = self.norm2(latents)

        return latents


class LatentsDecompressor(nn.Module):
    def __init__(self,
                 d_model=config["model"]["d_model"],
                 seq_len=config["model"]["seq_len"],
                 n_latents=config["model"]["total_seq_len"] // config["model"]["seq_len"],
                 z_i_dim=config["model"]["z_i_dim"],
                 z_tot_dim=config["model"]["z_tot_dim"],
                 token_reduction=4,
                 sequence_reduction=4,
                 latent_reduction=4,
                 instrument_reduction=4
                 ):
        super(LatentsDecompressor, self).__init__()
        # self.z_tot_dim = z_tot_dim
        # self.z_i_dim = z_i_dim
        self.n_latents = n_latents
        self.seq_len = seq_len
        self.d_model = d_model
        #
        # dim_1 = d_model // token_reduction
        # dim_2 = dim_1*seq_len // sequence_reduction
        # dim_3 = dim_2*n_latents // latent_reduction
        # dim_4 = dim_3*4 // instrument_reduction
        # self.decompress_instruments = nn.Linear(dim_4, dim_3*4)
        # self.decompress_track = nn.Linear(dim_3, dim_2*n_latents)
        # self.decompress_sequence = nn.Linear(dim_2, dim_1*seq_len)
        # self.decompress_token = nn.Linear(dim_1, d_model)
        #
        # self.norm3 = nn.LayerNorm(dim_3)
        # self.norm2 = nn.LayerNorm(dim_2)
        # self.norm1 = nn.LayerNorm(dim_1)
        # self.norm0 = nn.LayerNorm(d_model)
        self.decomp1 = nn.Linear(4*n_latents*d_model, 4*n_latents*d_model)
        self.norm1 = nn.LayerNorm(n_latents*d_model)
        self.decomp2 = nn.Linear(n_latents*d_model, n_latents*d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, latents):
        n_batch = latents.shape[0]
        # final_dim = (n_batch, 4, self.n_latents, self.seq_len, self.d_model)
        # latents = self.decompress_instruments(latents)
        # latents = latents.reshape(*final_dim[:2], -1)
        # latents = self.norm3(latents)
        # latents = self.decompress_track(latents)
        # latents = latents.reshape(*final_dim[:3], -1)
        # latents = self.norm2(latents)
        # latents = self.decompress_sequence(latents)
        # latents = latents.reshape(*final_dim[:4], -1)
        # latents = self.norm1(latents)
        # latents = self.decompress_token(latents)
        # latents = latents.reshape(*final_dim)
        # latents = self.norm0(latents)

        # latents = self.decomp1(latents)
        # latents = latents.reshape(n_batch, 4, -1)
        # latents = self.norm1(latents)
        # latents = self.decomp2(latents)
        # latents = latents.reshape(n_batch, 4, self.n_latents, -1)
        # latents = self.norm2(latents)
        latents = latents.unsqueeze(-2)
        return latents
