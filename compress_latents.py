import torch.nn as nn


class CompressLatents(nn.Module):
    def __init__(self, in_dim=None, z_dim=None, n_bars=None):
        super(CompressLatents, self).__init__()
        self.compress = nn.Linear(in_dim*n_bars, z_dim)

    def forward(self, x):
        x = x.transpose(0, 1)
        n_batch = x.shape[0]
        x = x.reshape(n_batch, -1)
        x = self.compress(x)
        return x


class DecompressLatents(nn.Module):
    def __init__(self, in_dim=None, z_dim=None, n_bars=None):
        super(DecompressLatents, self).__init__()
        self.decompress = nn.Linear(z_dim, in_dim*n_bars)
        self.n_bars = n_bars

    def forward(self, x):
        n_batch = x.shape[0]
        x = self.decompress(x)
        x = x.reshape(n_batch, self.n_bars, -1)
        x = x.transpose(0, 1)
        return x


# class CmemCompressor(nn.Module):
#     def __init__(self, d_model, final_dim, n_bars, n_layer):
#         super(CmemCompressor, self).__init__()
#         self.compress_layer = nn.Linear(d_model*n_layer, d_model)
#         self.compress_track = nn.Linear(4*d_model, d_model)
#         self.compress_bar = nn.Linear(n_bars*d_model, final_dim)
#
#     def forward(self, x):
#         n_track, n_layer, n_batch, n_bar, d_model = x.shape
#         x = x.transpose(1, 2)
#         x = x.reshape(n_track, n_batch, n_bar, -1)  # join layer
#         x = self.compress_layer(x)
#         x = x.reshape(n_batch, n_bar, -1)
#         x = self.compress_track(x)
#         x = x.reshape(n_batch, -1)
#         x = self.compress_bar(x)
#         return x
#
#
# class CmemDecompressor(nn.Module):
#     def __init__(self, d_model, final_dim, n_bars, n_layer):
#         super(CmemDecompressor, self).__init__()
#         self.decompress_layer = nn.Linear(d_model, d_model*n_layer)
#         self.decompress_track = nn.Linear(d_model, 4*d_model)
#         self.decompress_bar = nn.Linear(final_dim, n_bars*d_model)
#         self.n_bars = n_bars
#         self.n_layer = n_layer
#
#     def forward(self, x):
#         # n_track, n_layer, n_batch, n_bar, d_model = x.shape
#
#         n_batch = x.shape[0]
#         x = self.decompress_bar(x)
#         x = x.reshape(n_batch, self.n_bars, -1)
#         x = self.decompress_track(x)
#         x = x.reshape(4, n_batch, self.n_bars, -1)
#         x = self.decompress_layer(x)
#         x = x.reshape(4, n_batch, self.n_layer, self.n_bars, -1)
#         x = x.transpose(1, 2)
#         return x
