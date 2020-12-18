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
        y_drums = y[0, :, :]
        y_bass = y[1, :, :]
        y_guitar = y[2, :, :]
        y_strings = y[3, :, :]

        x_drums = x[:, :, :, 0]
        x_bass = x[:, :, :, 1]
        x_guitar = x[:, :, :, 2]
        x_strings = x[:, :, :, 3]

        loss_drums = self.smooth_label(x_drums.contiguous().view(-1, x_drums.size(-1)),
                                       y_drums.contiguous().view(-1))
        loss_bass = self.smooth_label(x_bass.contiguous().view(-1, x_bass.size(-1)),
                                      y_bass.contiguous().view(-1))
        loss_guitar = self.smooth_label(x_guitar.contiguous().view(-1, x_guitar.size(-1)),
                                        y_guitar.contiguous().view(-1))
        loss_strings = self.smooth_label(x_strings.contiguous().view(-1, x_strings.size(-1)),
                                         y_strings.contiguous().view(-1))

        loss = (loss_drums + loss_bass + loss_guitar + loss_strings) / norm[0]  # mean loss per token
        loss = loss.mean(dim=0)  # mean loss per batch sample

        loss_items = (loss_drums.item()/norm[1] if norm[1] != 0 else -1,
                      loss_bass.item()/norm[2] if norm[2] != 0 else -1,
                      loss_guitar.item()/norm[3] if norm[3] != 0 else -1,
                      loss_strings.item()/norm[4]if norm[4] != 0 else -1)

        return loss, loss_items
