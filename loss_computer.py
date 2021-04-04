import torch
from config import config
from torch import autograd
from torch import nn
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
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


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, enc_opt=None, dec_opt=None):
        self.generator = generator
        self.criterion = criterion
        self.enc_opt = enc_opt
        self.dec_opt = dec_opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        n_batch, n_track, seq_len, d_model = x.shape
        x = x.reshape(n_batch, n_track*seq_len, d_model)
        y = y.reshape(n_batch, n_track*seq_len)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        if self.generator.training:
            loss.backward()
            if self.enc_opt is not None:
                self.enc_opt.step()
                self.enc_opt.optimizer.zero_grad()
            if self.dec_opt is not None:
                self.dec_opt.step()
                self.dec_opt.optimizer.zero_grad()
        # compute accuracy
        pad_mask = y != self.criterion.padding_idx
        accuracy = ((torch.max(x, dim=-1).indices == y) & pad_mask).sum().item()
        accuracy = accuracy / pad_mask.sum().item()
        return loss.item(), accuracy  #  * norm, accuracy


def compute_accuracy(x, y, pad):  # TODO remove pad
    assert x.shape == y.shape
    y_pad = y != pad
    true = ((x == y) & y_pad).sum()
    count = y_pad.sum().item()
    return true/count


def calc_gradient_penalty(model, real_data, gen_data):
    device = config["train"]["device"]
    batch_size = config["train"]["batch_size"]
    alpha = torch.rand(batch_size, 1)

    alpha = alpha.expand(real_data.size()).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * gen_data)
    interpolates = autograd.Variable(interpolates.to(device), requires_grad=True)
    score_interpolates = model(interpolates)

    gradients = autograd.grad(
        outputs=score_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(score_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
