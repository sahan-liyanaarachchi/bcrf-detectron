from typing import Iterable

import torch
import torch.nn as nn

from crf.params import DenseCRFParams
from pytorch_permuto import compatibility_utils as cu
from pytorch_permuto.filters import SpatialFilter, BilateralFilter


def _apply_compatibility_transform(q_values: torch.Tensor, transform_matrix: torch.Tensor):
    c_in, h, w = q_values.shape
    _, c_out = transform_matrix.shape
    return torch.mm(transform_matrix.t(), q_values.view(c_in, -1)).view(c_out, h, w)


class PyTorchBCRF(nn.Module):
    """
    PyTorch implementation of Bipartite CRF
    """

    def __init__(
            self,
            sem_params: DenseCRFParams,
            ins_params: DenseCRFParams,
            num_labels: int,
            stuff_labels: Iterable[int],
            thing_labels: Iterable[int],
            num_iterations: int
    ):
        """
        Create a new instance

        Args:
            sem_params:     Semantic CRF parameters
            ins_params:     Instance CRF parameters
            num_labels:     Number of (semantic) labels in the dataset
            stuff_labels:   Stuff label IDs in the dataset
            num_iterations: Number of mean-field iterations
        """
        super().__init__()
        self.sem_params = sem_params
        self.ins_params = ins_params
        self.num_iterations = num_iterations

        self._softmax = torch.nn.Softmax(dim=0)  # TODO(sadeep):Can we reuse like this? Can it cause issues in backprop?

        self.stuff_labels = stuff_labels
        self.thing_labels = thing_labels

        # --------------------------------------------------------------------------------------------
        # ---------------------------------------- Parameters ----------------------------------------
        # --------------------------------------------------------------------------------------------

        # Class-specific spacial kernel weights for semantic segmentation
        self.param_sem_spatial_weights = nn.Parameter(
            sem_params.spatial_ker_weight * torch.ones(len(stuff_labels) + 1, dtype=torch.float32, device='cpu')
        )

        # Class-specific bilateral kernel weights for semantic segmentation
        self.param_sem_bilateral_weighs = nn.Parameter(
            sem_params.bilateral_ker_weight * torch.ones(len(stuff_labels) + 1, dtype=torch.float32, device='cpu')
        )

        # Compatibility transform weights for semantic segmentation
        self.param_sem_compatibility = nn.Parameter(
            torch.eye(len(stuff_labels)+1, dtype=torch.float32, device='cpu')
        )

        # Class-specific spacial kernel weights for instance segmentation
        self.param_ins_spatial_weights = nn.Parameter(
            sem_params.spatial_ker_weight * torch.ones(len(thing_labels) + 1, dtype=torch.float32, device='cpu')
        )

        # Class-specific bilateral kernel weights for instance segmentation
        self.param_ins_bilateral_weights = nn.Parameter(
            sem_params.spatial_ker_weight * torch.ones(len(thing_labels) + 1, dtype=torch.float32, device='cpu')
        )

        # TODO(sadeep) - Do we need a class-specific compatibility parameters for instances?

        # Cross compatibility parameters (instance to semantic)
        self.param_cross_ins_sem = nn.Parameter(
            cu.initial_cross_compatibility(thing_labels, stuff_labels)
        )

        # Cross compatibility parameters (semantic to instance)
        self.param_cross_sem_ins = nn.Parameter(
            cu.initial_cross_compatibility(thing_labels, stuff_labels)
        )

    def forward(self, image, semantic_logits, instance_logits, instance_cls_labels):
        """
        Perform BCRF inference.

        Args:
            image:                  Tensor of shape (3, h, w)
            semantic_logits:        Tensor of shape (num_sem_classes, h, w)
            instance_logits:        Tensor of shape (num_instances, h, w)
            instance_cls_labels:    Tensor of shape (num_instances,)
        Returns:
            (semantic_q_distributions, instance_q_distributions) after BCRF inference

        """
        sem_q = self._softmax(semantic_logits)
        ins_q = self._softmax(instance_logits)

        sem_spatial = SpatialFilter(image, gamma=self.sem_params.gamma)
        sem_bilateral = BilateralFilter(image, alpha=self.sem_params.alpha, beta=self.sem_params.beta)

        ins_spatial = SpatialFilter(image, gamma=self.ins_params.gamma)
        ins_bilateral = BilateralFilter(image, alpha=self.ins_params.alpha, beta=self.ins_params.beta)

        for _ in range(self.num_iterations):
            # Filter semantic segmentation Q distributions
            spatial_sem_out = self.param_sem_spatial_weights.view(len(self.stuff_labels) + 1, 1, 1).to(
                sem_q.device) * sem_spatial.apply(
                sem_q)
            bilateral_sem_out = self.param_sem_bilateral_weighs.view(len(self.stuff_labels) + 1, 1,
                                                                     1).to(sem_q.device) * sem_bilateral.apply(sem_q)
            tmp_sem = semantic_logits + _apply_compatibility_transform(
                spatial_sem_out + bilateral_sem_out,
                self.param_sem_compatibility.to(sem_q.device)
            )

            # Filter instance segmentation Q distributions
            spatial_weights = torch.index_select(
                self.param_ins_spatial_weights,
                dim=0,
                index=instance_cls_labels
            ).view(-1, 1, 1)
            spatial_ins_out = spatial_weights.to(sem_q.device) * ins_spatial.apply(ins_q)

            bilateral_weights = torch.index_select(
                self.param_ins_bilateral_weights,
                dim=0,
                index=instance_cls_labels
            ).view(-1, 1, 1)
            bilateral_ins_out = bilateral_weights.to(sem_q.device) * ins_bilateral.apply(ins_q)

            tmp_ins = instance_logits + spatial_ins_out + bilateral_ins_out

            # Cross potentials, instance to semantic
            ins_to_sem_compatibility = cu.get_compatibility(self.param_cross_ins_sem, instance_cls_labels)
            temp_sem_new = tmp_sem + _apply_compatibility_transform(
                self._softmax(tmp_ins),
                ins_to_sem_compatibility.to(sem_q.device)
            )

            # Cross potentials, semantic to instance
            sem_to_ins_compatibility = cu.get_compatibility(self.param_cross_sem_ins, instance_cls_labels).t()
            temp_ins_new = tmp_ins + _apply_compatibility_transform(
                self._softmax(tmp_sem),
                sem_to_ins_compatibility.to(sem_q.device)
            )

            # Normalize all distributions
            sem_q = self._softmax(temp_sem_new)
            ins_q = self._softmax(temp_ins_new)

        return sem_q, ins_q

#  TODO(HARSHA) is reset_parameters method needed?
