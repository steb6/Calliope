from torch import nn
import torch.nn.functional as F
# https://github.com/bfarzin/pytorch_aae


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, N, z_dim, dropout):
        super(self.dropout, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=self.dropout, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=self.dropout, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))
