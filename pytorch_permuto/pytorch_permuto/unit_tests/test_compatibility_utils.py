import unittest

import numpy as np
import torch

from crf import compatibility_transform as ct
from datasets.cityscapes import util as ds_util
from pytorch_permuto import compatibility_utils as cu


class TestPairwiseFeatureCalc(unittest.TestCase):

    def test_compat(self):
        num_labels = 19
        sem_labels = [-1, 11, 11, 13, 13, 11]

        np_mat = ct.ins_to_sem_compatibility(sem_labels, num_labels, stuff_sem_cls_ids=ds_util.STUFF_CLASS_IDS)

        param_matrix = cu.initial_cross_compatibility(num_labels, ds_util.STUFF_CLASS_IDS)
        sem_labels[0] = num_labels
        pt_instance_labels = torch.LongTensor(sem_labels)
        pt_mat = cu.get_compatibility(param_matrix, instance_cls_labels=pt_instance_labels)
        pt_mat = pt_mat.numpy()

        np.testing.assert_allclose(np_mat, pt_mat)
        self.assertAlmostEqual(np.max(np.abs(np_mat - pt_mat)), 0.0)


if __name__ == '__main__':
    unittest.main()
