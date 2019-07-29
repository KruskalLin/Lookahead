import copy


class Lookahead:
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.counter = 0
        self.param_groups = copy.deepcopy(optimizer.param_groups)

    def step(self):
        self.optimizer.step()
        self.counter += 1
        if self.counter == self.k:
            for i, group in enumerate(self.param_groups):
                for j, p in enumerate(group['params']):
                    p.data.copy_(self.param_groups[i]['params'][j].data +
                                (p.data - self.param_groups[i]['params'][j].data) * self.alpha)
                    self.param_groups[i]['params'][j].data.copy_(p.data)
            self.counter = 0

    def zero_grad(self):
        self.optimizer.zero_grad()