from collections import defaultdict
from torch.optim import Optimizer, Adam, SGD
import torch
import copy


class Lookahead(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, k=5, alpha=1.0):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.defaults['k'] = k
        self.defaults['alpha'] = alpha
        self.slow_dict = copy.deepcopy(self.param_groups)

    def update(self):
        for i in range(self.defaults['k']):
            self.step()
        for i, group in enumerate(self.slow_dict):
            for j, p in enumerate(group['params']):
                p.data.copy_(self.param_groups[i]['params'][j].data +
                             (self.param_groups[i]['params'][j].data - p.data) * self.defaults['alpha'])
                self.param_groups[i]['params'][j].data.copy_(p.data)