class CTOpt:
    """
    Optimizer wrapper that implements rate.
    """

    def __init__(self, optimizer, warmup_steps, warmup_interval):
        self.optimizer = optimizer
        self.step = 0
        self.warmup_steps = warmup_steps
        self.warmup_interval = warmup_interval

    def zero_grad(self):
        self.optimizer.zero_grad()

    def optimize(self):
        if self.step < self.warmup_steps:
            lr = (self.warmup_interval[1] - self.warmup_interval[0]) * (self.step / self.warmup_steps)\
                 + self.warmup_interval[0]
        else:
            lr = self.warmup_interval[1]
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.step += 1
        self.optimizer.step()
