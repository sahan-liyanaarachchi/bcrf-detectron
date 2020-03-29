import unittest

import numpy as np
import torch

from crf.pairwise import BilateralPairwise as NpBl
from crf.pairwise import SpatialPairwise as NpSp
from pytorch_permuto.filters import BilateralFilter as TorchBl
from pytorch_permuto.filters import SpatialFilter as TorchSp


class TestPairwiseFeatureCalc(unittest.TestCase):
    def setUp(self) -> None:
        h, w = 10, 20
        self.image = np.random.randn(h, w, 3).astype(np.float32)
        self.image_t = torch.from_numpy(np.transpose(self.image, (2, 0, 1)))

    def test_spatial_feature_calc(self):
        gamma = 5
        npsp = NpSp(self.image, sx=gamma, sy=gamma)
        a = npsp.features

        torchsp = TorchSp(self.image_t, gamma)
        b = torchsp.features.numpy()

        self.assertAlmostEqual(np.max(np.abs(a - b)), 0)
        np.testing.assert_allclose(a, b)

    def test_bilateral_feature_calc(self):
        alpha, beta = 3., 4.
        npbl = NpBl(self.image, sx=alpha, sy=alpha, sr=beta, sg=beta, sb=beta)
        a = npbl.features

        torchbl = TorchBl(self.image_t, alpha, beta)
        b = torchbl.features.numpy()

        self.assertAlmostEqual(np.max(np.abs(a - b)), 0)
        np.testing.assert_allclose(a, b)


if __name__ == '__main__':
    unittest.main()
