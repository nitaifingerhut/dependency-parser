import torch

from typing import Any


def to_device(x: Any, index: int = 0, dtype=torch.float64):
    if torch.cuda.is_available():
        x = x.cuda(index)
    x = x.type(dtype)
    return x


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()
