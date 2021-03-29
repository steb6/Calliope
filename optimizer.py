import numpy as np
import matplotlib.pyplot as plt
from config import config


class CTOpt:
    """
    Optimizer wrapper that implements rate.
    """

    def __init__(self, optimizer, warmup_steps, warmup_interval, decay_steps, minimum):
        print("OPTIMIZERS HAVE FIXED VALUES")
        self.optimizer = optimizer
        self.n_step = 0
        self.warmup_steps = warmup_steps
        self.warmup_interval = warmup_interval
        self.gap = warmup_interval[1] - warmup_interval[0]
        self.decay_steps = decay_steps
        self.minimum = minimum
        self.lr = 0

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_lr(self):
        # if self.n_step < self.warmup_steps:
        #     lr = (self.warmup_interval[1] - self.warmup_interval[0]) * (self.n_step / self.warmup_steps)\
        #          + self.warmup_interval[0]
        # elif self.n_step < self.warmup_steps + self.decay_steps:
        #     lr = self.gap*((np.cos(((self.n_step-self.warmup_steps)*np.pi)/self.decay_steps)+1)/2)+self.minimum
        # else:
        #     lr = self.minimum
        # self.n_step += 1
        # return lr
        return config["train"]["lr"]

    def step(self, lr=None):
        if lr is None:
            self.lr = self.get_lr()
        else:
            self.lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr
        self.optimizer.step()


class NoamOpt:
    "Optim wrapper that implements rate."
    # TODO try this
    def __init__(self, optimizer, model_size=config["model"]["d_model"], factor=1, warmup=0):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.lr = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.lr = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


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
