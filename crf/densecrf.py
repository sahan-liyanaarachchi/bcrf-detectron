from crf.pairwise import SpatialPairwise, BilateralPairwise
from crf.params import DenseCRFParams
from crf.util import softmax


class DenseCRF(object):

    def __init__(self, image, params: DenseCRFParams):
        alpha, beta, gamma = params.alpha, params.beta, params.gamma

        self.sp = SpatialPairwise(image, gamma, gamma)
        self.bp = BilateralPairwise(image, alpha, alpha, beta, beta, beta)

        self.spatial_weight = params.spatial_ker_weight
        self.bilateral_weight = params.bilateral_ker_weight

    def infer(self, unary_logits, num_iterations=5):
        q = softmax(unary_logits)

        for _ in range(num_iterations):
            tmp1 = unary_logits

            output = self.sp.apply(q)
            tmp1 = tmp1 + self.spatial_weight * output  # Do NOT use the += operator here!

            output = self.bp.apply(q)
            tmp1 = tmp1 + self.bilateral_weight * output

            q = softmax(tmp1)

        return q
