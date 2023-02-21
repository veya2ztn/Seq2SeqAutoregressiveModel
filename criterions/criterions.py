import torch
import torch.nn as nn
import numpy as np
class CenterWeightMSE(nn.Module):
    def __init__(self, center_range, boundary):
        super().__init__()
        self.boundary = boundary
        self.center_range = center_range
        center_x = (boundary-1)//2
        center_y = (boundary-1)//2
        weight = torch.zeros(boundary, boundary)
        for i in range(boundary):
            for j in range(boundary):
                weight[i, j] = np.sqrt(
                    (i - center_x)**2 + (j - center_y)**2)/(10.0*center_range/boundary)
        self.weight = weight.reshape(1, 1, boundary, boundary)

    def forward(self, pred, real):
        if real.shape[-2:] != (self.boundary, self.boundary):
            return torch.mean((pred - real)**2)
        else:
            return torch.mean(((pred-real)*self.weight.to(pred.device))**2)


class PressureWeightMSE(nn.Module):
    def __init__(self, alpha=0.5, min_weight=0.1, level=14):
        super().__init__()
        self.alpha = alpha
        self.min_weight = min_weight
        self.level = level
        self.weight = 1-torch.exp(-alpha*torch.arange(level))+min_weight
        self.weight = self.weight/self.weight.sum()*level
        self.weight = self.weight.reshape(1, 1, level, 1, 1)

    def forward(self, pred, real):

        delta = (pred - real)**2
        B, P, W, H = delta.shape
        delta = delta.reshape(B, -1, self.level, W, H) * \
            self.weight.to(pred.device)
        return delta.mean()
