import torch
from torch import nn


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing and apply given criterion.
    """

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
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        # unsqueeze add a dimension in the specified position
        # put self.confidence value (0.9) in true dist in the positions indicated by target
        # so true_dist has the same size of x which have 0.9 in the right notes and a small value in all the others
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # set all padding position to 0 (because we have a lot of them)
        true_dist[:, self.padding_idx] = 0
        # return position of padding of target
        mask = torch.nonzero(target.data == self.padding_idx)
        # substitute position of pad in true_dist with 0
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.to(self.device))
