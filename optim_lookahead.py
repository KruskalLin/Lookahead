from collections import defaultdict
from torch.optim import Optimizer, Adam, SGD
import torch
import copy


class Lookahead:
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = copy.deepcopy(optimizer.param_groups)

    def step(self):
        for i in range(self.k):
            self.optimizer.step()
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                p.data.copy_(self.param_groups[i]['params'][j].data +
                             (p.data - self.param_groups[i]['params'][j].data) * self.alpha)
                self.param_groups[i]['params'][j].data.copy_(p.data)

    def zero_grad(self):
        self.optimizer.zero_grad()