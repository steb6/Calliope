import numpy as np
import matplotlib.pyplot as plt
from config import config


class CTOpt:
    """
    Optimizer wrapper that implements rate.
    """

    def __init__(self, optimizer, warmup_steps, warmup_interval, decay_steps, minimum):
        self.optimizer = optimizer
        self.step = 0
        self.warmup_steps = warmup_steps
        self.warmup_interval = warmup_interval
        self.decay_steps = decay_steps
        self.minimum = minimum
        self.lr = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        if self.step < self.warmup_steps:
            lr = (self.warmup_interval[1] - self.warmup_interval[0]) * (self.step / self.warmup_steps)\
                 + self.warmup_interval[0]
        elif self.step < self.warmup_steps + self.decay_steps:
            lr = self.warmup_interval[1]*((np.cos(((self.step-self.warmup_steps)*np.pi)/self.decay_steps)+1)/2)+self.minimum
        else:
            lr = self.minimum
        self.step += 1
        return lr

    def optimize(self):
        self.lr = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr
        self.optimizer.step()


if __name__ == "__main__":
    opt = CTOpt(None, config["train"]["warmup_steps"],
                (config["train"]["lr_min"], config["train"]["lr_max"]),
                config["train"]["decay_steps"],
                config["train"]["minimum_lr"])
    x = list(range(config["train"]["warmup_steps"]*2+config["train"]["decay_steps"]))
    y = []
    for xi in x:
        y.append(opt.get_lr())
    plt.plot(x, y)
    plt.show()
