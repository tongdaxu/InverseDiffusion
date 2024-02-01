import torch
import torch.nn as nn
from .distance import DistanceInterface


class DistanceRMSE(nn.Module, DistanceInterface):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_hat):
        return torch.linalg.norm(y - y_hat)
