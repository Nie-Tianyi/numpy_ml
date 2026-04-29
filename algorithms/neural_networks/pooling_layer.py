"""
池化层
"""

import unittest

import numba
import numpy as np

from algorithms.neural_networks.neural_network_layer_abstract import NeuralNetworkLayerAbstract


class MaxPoolingLayer(NeuralNetworkLayerAbstract):
    """
    最大池化层，对每个通道独立做最大池化
    """

    def __init__(self, input_channels, input_height, input_width, pool_size=2, stride=2):
        self.input_channels_val = input_channels
        self.input_height_val = input_height
        self.input_width_val = input_width
        self.pool_size = pool_size
        self.stride = stride

        self.output_height = (input_height - pool_size) // stride + 1
        self.output_width = (input_width - pool_size) // stride + 1
        num = input_channels * self.output_height * self.output_width

        super().__init__(num)
        self.max_h_idx = None
        self.max_w_idx = None
        self.inputs = None

    def init_weights_and_bias(self, dim):
        expected = self.input_channels_val * self.input_height_val * self.input_width_val
        assert dim == expected, f"输入维度不匹配：期望 {expected}，实际 {dim}"
        self.weights = np.array([0])
        self.bias = np.array([0])

    def forward(self, x):
        m = x.shape[0]
        x_reshaped = x.reshape(m, self.input_channels_val, self.input_height_val, self.input_width_val)
        self.inputs = x_reshaped

        out, self.max_h_idx, self.max_w_idx = _max_pool_forward(
            x_reshaped, self.pool_size, self.stride
        )
        return out.reshape(m, -1)

    def backward(self, error, learning_rate, no_activation_grad=False):
        m = error.shape[0]
        error_reshaped = error.reshape(
            m, self.input_channels_val, self.output_height, self.output_width
        )
        self.reg_loss = 0

        prev_error = _max_pool_backward(
            error_reshaped, self.max_h_idx, self.max_w_idx, self.pool_size, self.stride,
            self.input_height_val, self.input_width_val,
        )
        return prev_error.reshape(m, -1)


@numba.njit(parallel=True, fastmath=True)
def _max_pool_forward(x, pool_size, stride):
    m, c, h, w = x.shape
    oh = (h - pool_size) // stride + 1
    ow = (w - pool_size) // stride + 1

    out = np.zeros((m, c, oh, ow), dtype=x.dtype)
    max_h = np.zeros((m, c, oh, ow), dtype=np.int32)
    max_w = np.zeros((m, c, oh, ow), dtype=np.int32)

    for i in numba.prange(m):
        for ch in range(c):
            for oy in range(oh):
                for ox in range(ow):
                    y_start = oy * stride
                    x_start = ox * stride
                    max_val = x[i, ch, y_start, x_start]
                    best_ph, best_pw = 0, 0
                    for py in range(pool_size):
                        for px in range(pool_size):
                            val = x[i, ch, y_start + py, x_start + px]
                            if val > max_val:
                                max_val = val
                                best_ph, best_pw = py, px
                    out[i, ch, oy, ox] = max_val
                    max_h[i, ch, oy, ox] = best_ph
                    max_w[i, ch, oy, ox] = best_pw

    return out, max_h, max_w


@numba.njit(parallel=True, fastmath=True)
def _max_pool_backward(error, max_h, max_w, pool_size, stride, h, w):
    m, c, oh, ow = error.shape

    dx = np.zeros((m, c, h, w), dtype=error.dtype)

    for i in numba.prange(m):
        for ch in range(c):
            for oy in range(oh):
                for ox in range(ow):
                    ph = max_h[i, ch, oy, ox]
                    pw = max_w[i, ch, oy, ox]
                    dx[i, ch, oy * stride + ph, ox * stride + pw] += error[i, ch, oy, ox]

    return dx


class Unittest(unittest.TestCase):
    def test_forward_shape(self):
        np.random.seed(42)
        pool = MaxPoolingLayer(input_channels=3, input_height=8, input_width=8, pool_size=2, stride=2)
        pool.init_weights_and_bias(192)
        x = np.random.randn(2, 192)
        out = pool.forward(x)
        self.assertEqual(out.shape, (2, 3 * 4 * 4))

    def test_backward_shape(self):
        np.random.seed(42)
        pool = MaxPoolingLayer(input_channels=2, input_height=6, input_width=6, pool_size=2, stride=2)
        pool.init_weights_and_bias(72)
        x = np.random.randn(4, 72)
        pool.forward(x)
        error = np.random.randn(4, 2 * 3 * 3) * 0.1
        prev_error = pool.backward(error, learning_rate=0.01)
        self.assertEqual(prev_error.shape, (4, 72))

    def test_max_pool_values(self):
        pool = MaxPoolingLayer(input_channels=1, input_height=4, input_width=4, pool_size=2, stride=2)
        pool.init_weights_and_bias(16)
        x = np.array([[
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        ]], dtype=np.float64)
        out = pool.forward(x)
        expected = np.array([[6, 8, 14, 16]], dtype=np.float64)
        self.assertTrue(np.allclose(out, expected))

    def test_max_pool_stride_1(self):
        pool = MaxPoolingLayer(input_channels=1, input_height=4, input_width=4, pool_size=2, stride=1)
        pool.init_weights_and_bias(16)
        x = np.array([[
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        ]], dtype=np.float64)
        out = pool.forward(x)
        expected = np.array([[
            6, 7, 8,
            10, 11, 12,
            14, 15, 16,
        ]], dtype=np.float64)
        self.assertTrue(np.allclose(out, expected))

    def test_backward_routes_to_correct_position(self):
        """验证反向传播将梯度只传递到最大值位置"""
        np.random.seed(42)
        pool = MaxPoolingLayer(input_channels=1, input_height=4, input_width=4, pool_size=2, stride=2)
        pool.init_weights_and_bias(16)
        x = np.array([[
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        ]], dtype=np.float64)
        pool.forward(x)
        error = np.ones((1, 4), dtype=np.float64) * 10.0
        prev_error = pool.backward(error, learning_rate=0.01)

        # 最大值位置: 6 at (1,1), 8 at (1,3), 14 at (3,1), 16 at (3,3)
        expected_sum = np.sum(np.abs(prev_error))
        self.assertGreater(expected_sum, 0)
        # 只有4个位置应该有非零梯度
        grad_reshaped = prev_error.reshape(4, 4)
        nonzero_count = np.count_nonzero(grad_reshaped)
        self.assertEqual(nonzero_count, 4)
        # 验证梯度只在正确的位置
        self.assertAlmostEqual(grad_reshaped[1, 1], 10.0)
        self.assertAlmostEqual(grad_reshaped[1, 3], 10.0)
        self.assertAlmostEqual(grad_reshaped[3, 1], 10.0)
        self.assertAlmostEqual(grad_reshaped[3, 3], 10.0)


if __name__ == "__main__":
    unittest.main()
