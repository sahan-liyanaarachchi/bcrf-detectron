import unittest

import numpy as np
import torch  # Import torch first!
from crf.py_permutohedral import PyPermutohedral

try:
    import permuto_cpp
except ImportError as e:
    raise (e, 'Did you import `torch` first?')

from pytorch_permuto.filters import PermutoFunction
from torch.autograd import gradcheck


class TestPermutoOp(unittest.TestCase):
    def setUp(self) -> None:
        self.n_features = 5
        self.height = 5
        self.width = 4
        self.n_classes = 10

        self.q_in = torch.randn(self.n_classes, self.height, self.width)
        self.q_in_gradcheck = torch.randn(self.n_classes, self.height, self.width, requires_grad=True)
        self.grad_q_out = torch.randn(self.n_classes, self.height, self.width)
        self.features = torch.randn(self.height, self.width, self.n_features, requires_grad=False)

    def test_spatial_filtering(self):
        self.assertRaises(RuntimeError, permuto_cpp.forward, self.q_in.double(), self.features)
        self.assertRaises(RuntimeError, permuto_cpp.forward, self.q_in, self.features.double())
        self.assertRaises(RuntimeError, permuto_cpp.forward, self.q_in,
                          torch.randn(self.height + 1, self.width, self.n_features))

    def test_with_numpy_impl(self):
        q_in_np = self.q_in.numpy()
        q_in_np = np.ascontiguousarray(np.transpose(q_in_np, [1, 2, 0]))  # Put channels at the end
        np_out = np.zeros_like(q_in_np)
        lattice = PyPermutohedral()
        lattice.init(self.features.numpy(), num_dimensions=self.n_features, num_points=self.height * self.width)
        lattice.compute(np_out, q_in_np, self.n_classes, False)

        pytouch_out = permuto_cpp.forward(self.q_in, self.features)[0]
        pytorch_out = np.transpose(pytouch_out.numpy(), [1, 2, 0])

        self.assertAlmostEqual(np.max(np.abs(np_out - pytorch_out)), 0)
        np.testing.assert_allclose(pytorch_out, np_out)

    def test_spatial_filtering_backwards(self):
        self.assertRaises(RuntimeError, permuto_cpp.backward, self.grad_q_out.double(), self.features)
        self.assertRaises(RuntimeError, permuto_cpp.backward, self.grad_q_out, self.features.double())
        self.assertRaises(RuntimeError, permuto_cpp.backward, self.grad_q_out,
                          torch.randn(self.height + 1, self.width, self.n_features))

    def test_backwards_with_numpy_impl(self):
        grad_q_out_np = self.grad_q_out.numpy()
        grad_q_out_np = np.ascontiguousarray(np.transpose(grad_q_out_np, [1, 2, 0]))  # Put channels at the end
        grad_q_back_np = np.zeros_like(grad_q_out_np)

        lattice = PyPermutohedral()
        lattice.init(self.features.numpy(), num_dimensions=self.n_features, num_points=self.height * self.width)
        lattice.compute(grad_q_back_np, grad_q_out_np, self.n_classes, True)

        pytorch_grad_back = permuto_cpp.backward(self.grad_q_out, self.features)[0]
        pytorch_grad_back = np.transpose(pytorch_grad_back.numpy(), [1, 2, 0])

        self.assertAlmostEqual(np.max(np.abs(grad_q_back_np - pytorch_grad_back)), 0)
        np.testing.assert_allclose(pytorch_grad_back, grad_q_back_np)

    def test_op_with_autograd_gradcheck(self):
        """
        Testing for 32 bit machine precision i.e. eps=3e-4 (relative perturbation : sqrt(machine precision))
        """
        PermutoFunc = PermutoFunction.apply
        test = gradcheck(PermutoFunc, (self.q_in_gradcheck, self.features), eps=3e-4, atol=1e-3, rtol=1e-5,
                         raise_exception=True)
        self.assertTrue(test)


if __name__ == '__main__':
    unittest.main()
