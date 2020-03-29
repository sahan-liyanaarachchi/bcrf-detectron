import numpy as np

from crf.densecrf import softmax
from crf.pairwise import SpatialPairwise, BilateralPairwise
from crf.params import DenseCRFParams


class BipartiteCRF(object):

    def __init__(self, image, sem_params: DenseCRFParams, ins_params: DenseCRFParams,
                 ins_to_sem_compatibility, sem_to_ins_compatibility):
        """
        Construct a bipartite-CRF for simultaneous semantic and instance segmentation. In the following L is the
        number of semantic labels and num_objs is the number of object detections (instances).

        Args:
            image:                      RGB image
            sem_params:                 Parameters for the semantic segmentation CRF
            ins_params:                 Parameters for the instance segmentation CRF
            ins_to_sem_compatibility:   A (num_objs x L) matrix
            sem_to_ins_compatibility:   An (L x num_objs) matrix
        """
        # Semantic segmentation pairwise potentials
        self.sem_sp = SpatialPairwise(image, sem_params.gamma, sem_params.gamma)
        self.sem_bp = BilateralPairwise(image, sem_params.alpha, sem_params.alpha,
                                        sem_params.beta, sem_params.beta, sem_params.beta)

        self.sem_spatial_weight = sem_params.spatial_ker_weight
        self.sem_bilateral_weight = sem_params.bilateral_ker_weight

        # Instance segmentation pairwise potentials
        self.inst_sp = SpatialPairwise(image, ins_params.gamma, ins_params.gamma)
        self.ins_bp = BilateralPairwise(image, ins_params.alpha, ins_params.alpha,
                                        ins_params.beta, ins_params.beta, ins_params.beta)

        self.ins_spatial_weight = ins_params.spatial_ker_weight
        self.ins_bilateral_weight = ins_params.bilateral_ker_weight

        self.ins_to_sem_compatibility = ins_to_sem_compatibility
        self.sem_to_ins_compatibility = sem_to_ins_compatibility

    def infer(self, sem_logits, ins_logits, num_iterations=5):
        sem_q = softmax(sem_logits)
        ins_q = softmax(ins_logits)

        for _ in range(num_iterations):
            tmp_sem = sem_logits
            tmp_ins = ins_logits

            # Filtering semantic segmentation Q distributions
            spatial_sem_out = self.sem_sp.apply(sem_q)
            tmp_sem = tmp_sem + self.sem_spatial_weight * spatial_sem_out  # Do NOT use the += operator here!

            bilateral_sem_out = self.sem_bp.apply(sem_q)
            tmp_sem = tmp_sem + self.sem_bilateral_weight * bilateral_sem_out

            # Filtering instance segmentation Q distributions
            spatial_ins_out = self.inst_sp.apply(ins_q)
            tmp_ins = tmp_ins + self.ins_spatial_weight * spatial_ins_out  # Do NOT use the += operator here!

            bilateral_ins_out = self.ins_bp.apply(ins_q)
            tmp_ins = tmp_ins + self.ins_bilateral_weight * bilateral_ins_out

            # Cross potentials
            temp_sem_new = tmp_sem + np.matmul(softmax(tmp_ins), self.ins_to_sem_compatibility)
            temp_ins_new = tmp_ins + np.matmul(softmax(tmp_sem), self.sem_to_ins_compatibility)

            # Normalize
            sem_q = softmax(temp_sem_new)
            ins_q = softmax(temp_ins_new)

        return sem_q, ins_q
