from torchmetrics import Metric
import torch
class MeanDistanceError(Metric):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_distance_errors", torch.tensor(0.0), dist_reduce_fx = "sum")
        self.add_state("n_observations", torch.tensor(0), dist_reduce_fx = "sum")

    def update(self, preds, target):
        error = torch.sum(torch.sqrt(torch.sum((preds - target)**2, axis = 1)))
        if error < 1e10:
            self.sum_distance_errors += error
            self.n_observations += preds.shape[0]
        else:
            self.n_observations += 1
    def compute(self):
        return self.sum_distance_errors / self.n_observations

if __name__ == "__main__":
    a = MeanDistanceError()
    preds = torch.randn(2,2) * 2e50
    target =  torch.zeros((2,2))
    a(preds,target)
    print(preds,a(preds,target))