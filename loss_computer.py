import torch


class SimpleLossCompute:

    def __init__(self, smooth_label):
        self.smooth_label = smooth_label

    def __call__(self, x, y, norm):
        """

        :param x: computed song with shape (n_batch, time-steps, vocab_dim, n_tracks)
        :param y: real song with shape (n_batch, timesteps, n_tracks)
        :param norm: Tuple with the total number of token and 4 respective number of token w.r.t. instruments
        :return: loss to use for back-propagation and instruments losses for plotting
        """
        n_bar, n_batch, n_tok, vocab_dim, n_track = x.shape
        x = x.reshape(n_batch, -1, vocab_dim, n_track)  # flat bars
        y = y.reshape(n_batch, -1, n_track)  # flat bars

        y_drums = y[..., 0]
        y_bass = y[..., 1]
        y_guitar = y[..., 2]
        y_strings = y[..., 3]

        x_drums = x[..., 0]
        x_bass = x[..., 1]
        x_guitar = x[..., 2]
        x_strings = x[..., 3]

        n_tokens_drums = torch.count_nonzero(y_drums).item()
        n_tokens_bass = torch.count_nonzero(y_bass).item()
        n_tokens_guitar = torch.count_nonzero(y_guitar).item()
        n_tokens_strings = torch.count_nonzero(y_strings).item()

        loss_drums = self.smooth_label(x_drums, y_drums) / n_tokens_drums
        loss_bass = self.smooth_label(x_bass, y_bass) / n_tokens_bass
        loss_guitar = self.smooth_label(x_guitar, y_guitar) / n_tokens_guitar
        loss_strings = self.smooth_label(x_strings, y_strings) / n_tokens_strings

        loss = (loss_drums + loss_bass + loss_guitar + loss_strings) / 4  # mean loss per token

        return loss, (loss_drums, loss_bass, loss_guitar, loss_strings)
