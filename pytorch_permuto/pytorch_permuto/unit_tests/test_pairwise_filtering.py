import unittest

import numpy as np
import torch

from crf.pairwise import BilateralPairwise as NpBl
from crf.pairwise import SpatialPairwise as NpSp
from pytorch_permuto.filters import BilateralFilter as TorchBl
from pytorch_permuto.filters import SpatialFilter as TorchSp


class TestPairwiseFiltering(unittest.TestCase):
    def setUp(self) -> None:
        h, w = 10, 20
        c = 6
        self.image = np.random.randn(h, w, 3).astype(np.float32)
        self.image_t = torch.from_numpy(np.transpose(self.image, (2, 0, 1)))

        self.q = np.random.randn(h, w, c).astype(np.float32)
        self.q_t = torch.from_numpy(np.transpose(self.q, (2, 0, 1)))

    def test_spatial_filtering(self):
        gamma = 5
        npsp = NpSp(self.image, sx=gamma, sy=gamma)
        np_norm = npsp.norm
        np_q_out = npsp.apply(self.q)

        torchsp = TorchSp(self.image_t, gamma)
        pt_norm = torchsp.norm.numpy()
        pt_q_out = torchsp.apply(self.q_t).numpy()
        pt_q_out = np.transpose(pt_q_out, (1, 2, 0))

        np.testing.assert_allclose(np_norm.flatten(), pt_norm.flatten())
        np.testing.assert_allclose(np_q_out, pt_q_out)

        self.assertAlmostEqual(np.max(np.abs(np_norm.flatten() - pt_norm.flatten())), 0)
        self.assertAlmostEqual(np.max(np.abs(np_q_out - pt_q_out)), 0)

    def test_bilateral_feature_calc(self):
        alpha, beta = 3., 4.
        npbl = NpBl(self.image, sx=alpha, sy=alpha, sr=beta, sg=beta, sb=beta)
        np_norm = npbl.norm
        np_q_out = npbl.apply(self.q)

        torchbl = TorchBl(self.image_t, alpha, beta)
        pt_norm = torchbl.norm.numpy()
        pt_q_out = torchbl.apply(self.q_t).numpy()
        pt_q_out = np.transpose(pt_q_out, (1, 2, 0))

        np.testing.assert_allclose(np_norm.flatten(), pt_norm.flatten())
        np.testing.assert_allclose(np_q_out, pt_q_out)

        self.assertAlmostEqual(np.max(np.abs(np_norm.flatten() - pt_norm.flatten())), 0)
        self.assertAlmostEqual(np.max(np.abs(np_q_out - pt_q_out)), 0)


if __name__ == '__main__':
    unittest.main()
