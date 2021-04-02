import torch
from config import config
from torch import autograd
from torch import nn
from torch.autograd import Variable


class SimpleLossCompute:

    def __init__(self, smooth_label):
        self.smooth_label = smooth_label
        self.pad = config["tokens"]["pad"]

    def __call__(self, x, y):
        """

        :param x: computed song with shape (n_batch, time-steps, vocab_dim, n_tracks)
        :param y: real song with shape (n_batch, timesteps, n_tracks)
        :return: loss to use for back-propagation and instruments losses for plotting
        """

        assert x.shape[:-1] == y.shape

        n_bars = float(x.shape[0])

        loss = torch.zeros(1, requires_grad=True, dtype=torch.float32, device=x.device)
        loss_drums_total = 0
        loss_guitar_total = 0
        loss_bass_total = 0
        loss_strings_total = 0

        for x_bar, y_bar in zip(x, y):
            y_drums = y_bar[0]
            y_guitar = y_bar[1]
            y_bass = y_bar[2]
            y_strings = y_bar[3]

            x_drums = x_bar[0]
            x_guitar = x_bar[1]
            x_bass = x_bar[2]
            x_strings = x_bar[3]

            n_tokens_drums = (y_drums != self.pad).sum()
            n_tokens_guitar = (y_guitar != self.pad).sum()
            n_tokens_bass = (y_bass != self.pad).sum()
            n_tokens_strings = (y_strings != self.pad).sum()

            loss_drums = self.smooth_label(x_drums, y_drums) / n_tokens_drums
            loss_guitar = self.smooth_label(x_guitar, y_guitar) / n_tokens_guitar
            loss_bass = self.smooth_label(x_bass, y_bass) / n_tokens_bass
            loss_strings = self.smooth_label(x_strings, y_strings) / n_tokens_strings

            loss = loss + ((loss_drums + loss_guitar + loss_bass + loss_strings) / 4)  # mean loss per token

            loss_drums_total += loss_drums.item()
            loss_guitar_total += loss_guitar.item()
            loss_bass_total += loss_bass.item()
            loss_strings_total += loss_strings.item()

        return loss/n_bars, (loss_drums_total/n_bars, loss_guitar_total/n_bars,
                             loss_bass_total/n_bars, loss_strings_total/n_bars)


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0, device=None):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")  # size_average=False
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.device = device

    def forward(self, x, target):
        # x = x[0]
        # target = target[0]  # TODO adjust for batch_size > 1
        assert x.size(2) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(2, target.data.unsqueeze(2), self.confidence)
        true_dist[:, :, self.padding_idx] = 0  # it was true_dist[:, self.padding_idx] = 0  # TODO CHECK BETTER
        mask = torch.nonzero(target.data == self.padding_idx)
        # if mask.dim() > 0:
        for elem in mask:
            true_dist[elem[0], elem[1], :] = 0
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


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
        # only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
