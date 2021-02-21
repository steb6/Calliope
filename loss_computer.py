import torch
from config import config
from torch import autograd


class SimpleLossCompute:

    def __init__(self, smooth_label):
        self.smooth_label = smooth_label

    def __call__(self, x, y):
        """

        :param x: computed song with shape (n_batch, time-steps, vocab_dim, n_tracks)
        :param y: real song with shape (n_batch, timesteps, n_tracks)
        :return: loss to use for back-propagation and instruments losses for plotting
        """
        # n_bar, n_batch, n_tok, vocab_dim, n_track = x.shape
        # x = x.reshape(n_batch, -1, vocab_dim, n_track)  # flat bars
        # y = y.reshape(n_batch, -1, n_track)  # flat bars

        y_drums = y[..., 0]
        y_guitar = y[..., 1]
        y_bass = y[..., 2]
        y_strings = y[..., 3]

        x_drums = x[..., 0]
        x_guitar = x[..., 1]
        x_bass = x[..., 2]
        x_strings = x[..., 3]

        n_tokens_drums = torch.count_nonzero(y_drums).item()
        n_tokens_guitar = torch.count_nonzero(y_guitar).item()
        n_tokens_bass = torch.count_nonzero(y_bass).item()
        n_tokens_strings = torch.count_nonzero(y_strings).item()

        loss_drums = self.smooth_label(x_drums, y_drums) / n_tokens_drums
        loss_guitar = self.smooth_label(x_guitar, y_guitar) / n_tokens_guitar
        loss_bass = self.smooth_label(x_bass, y_bass) / n_tokens_bass
        loss_strings = self.smooth_label(x_strings, y_strings) / n_tokens_strings

        loss = (loss_drums + loss_guitar + loss_bass + loss_strings) / 4  # mean loss per token

        return loss, (loss_drums.item(), loss_guitar.item(), loss_bass.item(), loss_strings.item())


def compute_accuracy(x, y, pad):
    x = torch.max(x, dim=-2).indices.transpose(1, 2)
    y = y.transpose(1, 2)
    true = 0
    count = 0
    for xi, yi in zip(x, y):
        for xij, yij in zip(xi, yi):
            for xijk, yijk in zip(xij, yij):
                if yijk != pad:
                    count += 1
                    if xijk == yijk:
                        true += 1
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

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
