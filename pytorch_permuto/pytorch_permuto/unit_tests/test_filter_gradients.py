import unittest

import numpy as np
import torch
from torch.autograd import gradcheck

from pytorch_permuto.filters import BilateralFilter as TorchBl
from pytorch_permuto.filters import SpatialFilter as TorchSp


class TestGradPairwiseFiltering(unittest.TestCase):
    def setUp(self) -> None:
        h, w = 10, 20
        c = 6
        self.image = np.random.randn(h, w, 3).astype(np.float32)
        self.image_t = torch.from_numpy(np.transpose(self.image, (2, 0, 1))).requires_grad_(False)

        self.q = np.random.randn(h, w, c).astype(np.float32)
        self.q_t = torch.from_numpy(np.transpose(self.q, (2, 0, 1))).requires_grad_(True)

    def test_grad_spatial_filtering(self):
        gamma = 5
        torchsp = TorchSp(self.image_t, gamma)
        self.assertTrue(gradcheck(torchsp.apply, (self.q_t,), eps=3e-4, atol=1e-3, rtol=1e-5, raise_exception=True))

    def test_grad_bilateral_feature_calc(self):
        alpha, beta = 3., 4.
        torchbl = TorchBl(self.image_t, alpha, beta)
        self.assertTrue(gradcheck(torchbl.apply, (self.q_t,), eps=3e-4, atol=1e-3, rtol=1e-5, raise_exception=True))


if __name__ == '__main__':
    unittest.main()
