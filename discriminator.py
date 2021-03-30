import torch
from torch import nn
import torch.nn.functional as F
from config import config
# https://github.com/bfarzin/pytorch_aae


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, d_model, dropout):  # seq_len, d_model, dropout
        super(Discriminator, self).__init__()

        latent_size = config["model"]["d_model"]

        self.lin1 = nn.Linear(latent_size, latent_size//4)
        self.norm1 = nn.LayerNorm(latent_size//4)
        self.lin2 = nn.Linear(latent_size//4, latent_size//8)
        self.norm2 = nn.LayerNorm(latent_size//8)
        self.lin3 = nn.Linear(latent_size//8, 1)
        self.dropout = dropout

    def forward(self, x):
        x = self.lin1(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)
        x = self.lin3(x)
        return x
