import os.path as osp
import unittest

import numpy as np
import torch
from PIL import Image
from torch.autograd import gradcheck

from UPSNet.serve.utils import DataKeys as dk
from crf import compatibility_transform as ct
from crf.bipartite_crf import BipartiteCRF
from crf.densecrf import DenseCRFParams
from datasets.cityscapes import util as ds_util
from pytorch_permuto.pytorch_bcrf import PyTorchBCRF
from pytorch_permuto.unit_tests.util import to_torch, to_numpy

_EPS = np.finfo('float').eps
work_dir = '/home/harsha/fyp/segmentation/bcrf/UPSNet/data'
im_title = 'dog_walking'
im_path = osp.join(work_dir, im_title + '.jpg')
arrays_path = osp.join(work_dir, im_title + '.npz')


class TestPyTorchBCRF(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_inference(self):
        image = Image.open(im_path)
        arrays = np.load(arrays_path)
        num_labels = 19
        sem_logits = arrays[dk.semantic_logits]
        ins_logits = np.log(arrays[dk.instance_probs] + _EPS)
        obj_labels = arrays[dk.instance_labels]
        sem_labels = ds_util.obj_det_ids_to_sem_ids(obj_labels)
        sem_labels = [-1, ] + sem_labels
        mat1 = ct.ins_to_sem_compatibility(sem_labels, num_labels, stuff_sem_cls_ids=ds_util.STUFF_CLASS_IDS)
        mat2 = mat1.T

        sem_params = DenseCRFParams(spatial_ker_weight=10, bilateral_ker_weight=10, alpha=150, beta=2, gamma=3)
        ins_params = DenseCRFParams(
            spatial_ker_weight=10,
            bilateral_ker_weight=10,
            alpha=150,
            beta=2,
            gamma=3
        )

        ins_to_sem_compt = mat1
        sem_to_ins_compt = mat2
        crf = BipartiteCRF(
            np.array(image),
            sem_params=sem_params,
            ins_params=ins_params,
            ins_to_sem_compatibility=ins_to_sem_compt,
            sem_to_ins_compatibility=sem_to_ins_compt
        )

        sem_q, ins_q = crf.infer(sem_logits, ins_logits)
        pt_bcrf = PyTorchBCRF(
            sem_params=sem_params, ins_params=ins_params,
            num_labels=19,
            stuff_labels=ds_util.STUFF_CLASS_IDS,
            num_iterations=5
        )

        pt_image = to_torch(np.array(image))
        pt_sem_logits = to_torch(sem_logits)
        pt_ins_logits = to_torch(ins_logits)
        sem_labels[0] = num_labels
        pt_instance_cls_labels = torch.LongTensor(sem_labels)

        pt_sem_q, pt_ins_q = pt_bcrf(pt_image, pt_sem_logits, pt_ins_logits, pt_instance_cls_labels)

        pt_sem_q = to_numpy(pt_sem_q)
        pt_ins_q = to_numpy(pt_ins_q)

        sem_diff = np.abs(sem_q - pt_sem_q)
        ins_diff = np.abs(ins_q - pt_ins_q)

        print('Max diffs are: sem = {} ({:.2f}%), ins = {} ({:.2f}%)'.format(
            np.max(sem_diff),
            np.max(sem_diff) / np.mean(np.abs(sem_q)) * 100,
            np.max(ins_diff),
            np.max(ins_diff) / np.mean(np.abs(ins_q)) * 100
        ))

        sem_q[sem_q < 1e-6] = 0
        pt_sem_q[pt_sem_q < 1e-6] = 0
        np.testing.assert_allclose(sem_q, pt_sem_q, atol=0, rtol=1e-3)

        ins_q[ins_q < 1e-6] = 0
        pt_ins_q[pt_ins_q < 1e-6] = 0
        np.testing.assert_allclose(ins_q, pt_ins_q, atol=0, rtol=1e-3)

    def test_gradients(self):
        image = Image.open(im_path)
        arrays = np.load(arrays_path)
        num_labels = 19
        sem_logits = arrays[dk.semantic_logits]
        ins_logits = np.log(arrays[dk.instance_probs] + _EPS)
        obj_labels = arrays[dk.instance_labels]
        sem_labels = ds_util.obj_det_ids_to_sem_ids(obj_labels)
        sem_labels = [-1, ] + sem_labels

        sem_params = DenseCRFParams(spatial_ker_weight=10, bilateral_ker_weight=10, alpha=150, beta=2, gamma=3)
        ins_params = DenseCRFParams(
            spatial_ker_weight=10,
            bilateral_ker_weight=10,
            alpha=150,
            beta=2,
            gamma=3
        )

        pt_bcrf = PyTorchBCRF(
            sem_params=sem_params, ins_params=ins_params,
            num_labels=19,
            stuff_labels=ds_util.STUFF_CLASS_IDS,
            num_iterations=5
        )

        pt_image = to_torch(np.array(image)).requires_grad_(False)
        pt_sem_logits = to_torch(sem_logits).requires_grad_(True)
        pt_ins_logits = to_torch(ins_logits).requires_grad_(True)
        sem_labels[0] = num_labels
        pt_instance_cls_labels = torch.LongTensor(sem_labels).requires_grad_(False)

        #  Cropping the image and logits
        pt_image = pt_image[:, 0:15, 0:25]
        pt_sem_logits = pt_sem_logits[:, 0:15, 0:25]
        pt_ins_logits = pt_ins_logits[:, 0:15, 0:25]

        self.assertTrue(gradcheck(pt_bcrf, (pt_image, pt_sem_logits, pt_ins_logits, pt_instance_cls_labels),
                                  eps=3e-4, atol=1e-3, rtol=1e-5, raise_exception=True))


if __name__ == '__main__':
    unittest.main()
