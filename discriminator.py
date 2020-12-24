import torch.nn as nn
import torch.nn.functional as f
import torch
# https://github.com/bfarzin/pytorch_aae


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, N, z_dim):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = f.dropout(self.lin1(x), p=0.2, training=self.training)
        x = f.relu(x)
        x = f.dropout(self.lin2(x), p=0.2, training=self.training)
        x = f.relu(x)
        return torch.sigmoid(self.lin3(x))

# # Encoder
# class Q_net(nn.Module):
#     def __init__(self, X_dim, N, z_dim):
#         super(Q_net, self).__init__()
#         self.lin1 = nn.Linear(X_dim, N)
#         self.lin2 = nn.Linear(N, N)
#         self.lin3gauss = nn.Linear(N, z_dim)
#
#     def forward(self, x):
#         x = F.dropout(self.lin1(x), p=0.25, training=self.training)
#         x = F.relu(x)
#         x = F.dropout(self.lin2(x), p=0.25, training=self.training)
#         x = F.relu(x)
#         xgauss = self.lin3gauss(x)
#         return xgauss
#
#
# # Decoder
# class P_net(nn.Module):
#     def __init__(self, X_dim, N, z_dim):
#         super(P_net, self).__init__()
#         self.lin1 = nn.Linear(z_dim, N)
#         self.lin2 = nn.Linear(N, N)
#         self.lin3 = nn.Linear(N, X_dim)
#
#     def forward(self, x):
#         x = F.dropout(self.lin1(x), p=0.25, training=self.training)
#         x = F.relu(x)
#         x = F.dropout(self.lin2(x), p=0.25, training=self.training)
#         x = self.lin3(x)
#         return torch.sigmoid(x)


# EPS = 1e-15
# # Reconstruction loss: optimize encoder Q and decoder P to reconstruct input
# self.P_net.zero_grad()
# self.Q_net.zero_grad()
# self.D_net_gauss.zero_grad()
# z = self.Q_net(latents)
# x = self.P_net(z)
# # recon_loss = F.binary_cross_entropy(x+EPS, latents+EPS)  TODO mse or binary cross entropy?
# recon_loss = F.mse_loss(x + EPS, latents + EPS)
# # recon_loss = recon_loss / (self.model.d_model*self.max_bars)
# recon_loss.backward()
# self.optim_P.step()
# self.optim_Q_enc.step()
# # Discriminator: optimize discriminator to recognize normal distribution as valid
# self.Q_net.eval()
# z_real_gauss = Variable(torch.randn_like(z)).cuda()  # normal distribution
# D_real_gauss = self.D_net_gauss(z_real_gauss)
# z_fake_gauss = self.Q_net(latents)
# D_fake_gauss = self.D_net_gauss(z_fake_gauss)
# D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
# D_loss.backward()
# self.optim_D.step()
# # Generator
# self.Q_net.train()
# z_fake_gauss = self.Q_net(latents)
# D_fake_gauss = self.D_net_gauss(z_fake_gauss)
# G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))
# G_loss.backward()
# self.optim_Q_gen.step()
