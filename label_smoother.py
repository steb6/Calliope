import torch
from torch import nn
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0, device=None):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.device = device

    def forward(self, x, target):
        x = x[0]
        target = target[0]  # TODO adjust for batch_size > 1
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


#
# class LabelSmoothing(nn.Module):
#     """
#     Implement label smoothing and apply given criterion.
#     """
#
#     def __init__(self, size, padding_idx, smoothing=0.0, device=None):
#         super(LabelSmoothing, self).__init__()
#         self.criterion = nn.KLDivLoss(size_average=False)
#         self.padding_idx = padding_idx
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.size = size
#         self.true_dist = None
#         self.device = device
#
#     def forward(self, x, target):
#         # assert x.size(1) == self.size
#         true_dist = x.data.clone()
#         true_dist.fill_(self.smoothing / (self.size - 2))
#         true_idx = target.data.unsqueeze(2)
#         # true_dist[i][j][aux[i][j][k]] = self.confidence
#         true_dist.scatter_(2, true_idx, self.confidence)
#         true_dist[..., self.padding_idx] = 0  # we dont want padding to be targeted
#         # return position of padding of target
#         mask = torch.nonzero(target.data == self.padding_idx)
#         # mask = (target.data == self.padding_idx).nonzero()
#         # mas = torch.nonzero(target.data, (target.data == self.padding_idx))
#         # substitute position of pad in true_dist with 0
#         for m in mask:
#             true_dist[m[0], m[1], :] = 0.0
#         # if mask.dim() > 0:
#         #     true_dist.index_fill_(0, mask.squeeze(), 0.0)
#         self.true_dist = true_dist
#         return self.criterion(x, true_dist.to(self.device))
