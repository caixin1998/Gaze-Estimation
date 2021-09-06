
from util.util import torch_angular_error

from torchmetrics import Metric
import torch
class MeanAngularError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_angular_errors", torch.tensor(0.0), dist_reduce_fx = "sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx = "sum")

    def update(self, preds, target):
        self.sum_angular_errors += torch_angular_error(preds, target)
        self.n_observations += preds.shape[0]
        # print(self.sum_angular_errors, self.n_observations)
    def compute(self):
        return self.sum_angular_errors / self.n_observations