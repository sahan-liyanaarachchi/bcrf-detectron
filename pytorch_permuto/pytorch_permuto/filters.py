from abc import ABC, abstractmethod

import numpy as np
import torch

try:
    import permuto_cpp
except ImportError as e:
    raise (e, 'Did you import `torch` first?')

_CPU = torch.device("cpu")
_EPS = np.finfo('float').eps


class PermutoFunction(torch.autograd.Function):
    """
    Usage:
        Use Function.apply method. Alias this as 'PermutoFunc'.
        PermutoFunc = permutoFunc.apply
    Forward pass: compute q_out from q_in, features
    """

    @staticmethod
    def forward(ctx, q_in: torch.Tensor, features: torch.Tensor):
        q_out = permuto_cpp.forward(q_in, features)[0]
        ctx.save_for_backward(features)
        return q_out

    @staticmethod
    def backward(ctx, grad_q_out: torch.Tensor):
        feature_saved = ctx.saved_tensors[0]
        grad_q_back = permuto_cpp.backward(grad_q_out.contiguous(), feature_saved.contiguous())[0]
        return grad_q_back, None  # No need of grads w.r.t. features


def _spatial_features(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Return the spatial features as a Tensor

    Args:
        image:
        sigma:

    Returns:
        Tensor of shape [h, w, 2] with spatial features
    """
    sigma = float(sigma)
    _, h, w = image.size()
    x = torch.arange(start=0, end=w, dtype=torch.float32, device=_CPU)
    xx = x.repeat([h, 1]) / sigma

    y = torch.arange(start=0, end=h, dtype=torch.float32, device=torch.device("cpu")).view(-1, 1)
    yy = y.repeat([1, w]) / sigma

    return torch.stack([xx, yy], dim=2)


class AbstractFilter(ABC):
    """
    Super-class for permutohedral-based Gaussian filters
    """

    def __init__(self, image: torch.Tensor):
        self.features = self._calc_features(image)
        self.norm = self._calc_norm(image)

    def apply(self, input_: torch.Tensor) -> torch.Tensor:
        input_ = input_ * self.norm
        output = PermutoFunction.apply(input_, self.features)
        return output * self.norm

    @abstractmethod
    def _calc_features(self, image: torch.Tensor) -> torch.Tensor:
        pass

    def _calc_norm(self, image: torch.Tensor) -> torch.Tensor:
        _, h, w = image.size()
        all_ones = torch.ones((1, h, w), dtype=torch.float32, device=_CPU)
        norm = PermutoFunction.apply(all_ones, self.features)
        return 1.0 / torch.sqrt(norm + _EPS)


class SpatialFilter(AbstractFilter):
    """
    Gaussian filter in the spatial ([x, y]) domain
    """

    def __init__(self, image: torch.Tensor, gamma):
        """
        Create new instance

        Args:
            image:  Image tensor of shape (3, height, width)
            gamma:  Standard deviation
        """
        self.gamma = gamma
        super().__init__(image)

    def _calc_features(self, image: torch.Tensor):
        return _spatial_features(image, self.gamma)


class BilateralFilter(AbstractFilter):
    """
    Gaussian filter in the bilateral ([r, g, b, x, y]) domain
    """

    def __init__(self, image: torch.Tensor, alpha: float, beta: float):
        """
        Create new instance

        Args:
            image:  Image tensor of shape (3, height, width)
            alpha:  Smoothness (spatial) sigma
            beta:   Appearance (color) sigma
        """
        self.alpha = alpha
        self.beta = beta
        super().__init__(image)

    def _calc_features(self, image: torch.Tensor):
        xy = _spatial_features(image, self.alpha)  # TODO Possible optimisation, was calculated in the spatial kernel
        rgb = (image / float(self.beta)).permute(1, 2, 0)  # Channel last order
        return torch.cat([xy, rgb], dim=2)
