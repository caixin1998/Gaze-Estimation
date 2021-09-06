from torchmetrics import Metric
import torch
class MeanDistanceError(Metric):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_distance_errors", torch.tensor(0.0), dist_reduce_fx = "sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx = "sum")

    def update(self, preds, target):
        self.sum_distance_errors += torch.sum(torch.sqrt(torch.sum((preds - target)**2, axis = 1)))
        self.n_observations += preds.shape[0]
    def compute(self):
        return self.sum_distance_errors / self.n_observations

if __name__ == "__main__":
    a = MDE()
    preds = torch.randn(2,2)
    target =  torch.zeros((2,2))
    a(preds,target)
    print(preds,a(preds,target))