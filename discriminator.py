import torch
from torch import nn
import torch.nn.functional as F
# https://github.com/bfarzin/pytorch_aae


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, d_model, dropout):  # seq_len, d_model, dropout
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(d_model, d_model//2)
        self.lin2 = nn.Linear(d_model//2, d_model//4)
        self.lin3 = nn.Linear(d_model//4, 1)
        self.dropout = dropout

    def forward(self, x):
        # x = x.reshape(x.shape[0], -1)
        x = F.dropout(self.lin1(x), p=self.dropout, training=self.training)
        x = F.leaky_relu(x)
        x = F.dropout(self.lin2(x), p=self.dropout, training=self.training)
        x = F.leaky_relu(x)
        return torch.sigmoid(self.lin3(x))
