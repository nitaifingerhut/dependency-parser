import torch
import torch.nn as nn

from typing import Optional
from utils.torch import to_device


class DependencyParserNLLLoss(nn.Module):

    def __init__(self, dim: Optional[int] = 1, ignore_index: Optional[int] = -1):
        super(DependencyParserNLLLoss, self).__init__()

        logsoftmax = torch.nn.LogSoftmax(dim=dim)
        self.logsoftmax = to_device(logsoftmax, dtype=torch.float64)

        nllloss = torch.nn.NLLLoss(ignore_index=ignore_index, reduction="mean")
        self.nllloss = to_device(nllloss, dtype=torch.float64)

    def __call__(self, scores: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        return self.nllloss(self.logsoftmax(scores), heads)
