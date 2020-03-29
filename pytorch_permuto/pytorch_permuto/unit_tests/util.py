import numpy as np
import torch


def to_torch(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.transpose([2, 0, 1]).astype(np.float32, copy=False))


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().numpy().transpose([1, 2, 0])
