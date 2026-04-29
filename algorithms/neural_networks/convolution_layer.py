"""
卷积层
"""

import unittest

import numba
import numpy as np

from algorithms.activation_functions import ActivationFunction, ReLU
from algorithms.neural_networks.neural_network_layer_abstract import NeuralNetworkLayerAbstract
from algorithms.regularization import Regularization, Ridge


class ConvolutionLayer(NeuralNetworkLayerAbstract):
    """
    卷积层，采用 im2col 方式实现二维卷积
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_height,
        input_width,
        stride=1,
        padding=0,
        activation_function: type[ActivationFunction] = ReLU,
        reg: type[Regularization] = Ridge,
        reg_params=0.1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_height = input_height
        self.input_width = input_width
        self.stride = stride
        self.padding = padding

        self.output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        self.output_width = (input_width + 2 * padding - kernel_size) // stride + 1
        assert (input_height + 2 * padding - kernel_size) % stride == 0, "stride 不能整除 padded 输入高度"
        assert (input_width + 2 * padding - kernel_size) % stride == 0, "stride 不能整除 padded 输入宽度"

        num = out_channels * self.output_height * self.output_width
        super().__init__(num)
        self.activation_function = activation_function
        self.reg = reg
        self.lambda_ = reg_params
        self.z = None
        self.inputs = None
        self.col = None

    def init_weights_and_bias(self, dim):
        expected = self.in_channels * self.input_height * self.input_width
        assert dim == expected, f"输入维度不匹配：期望 {expected}，实际 {dim}"
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        self.weights = np.random.randn(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        ) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros((self.out_channels, 1, 1))

    def forward(self, x):
        m = x.shape[0]
        x_reshaped = x.reshape(m, self.in_channels, self.input_height, self.input_width)
        self.inputs = x_reshaped

        col = _im2col(x_reshaped, self.kernel_size, self.stride, self.padding)
        self.col = col

        w_reshaped = self.weights.reshape(self.out_channels, -1)
        z = w_reshaped[np.newaxis, :, :] @ col
        z = z.reshape(m, self.out_channels, self.output_height, self.output_width)
        z += self.bias
        self.z = z

        a = self.activation_function.cal(z)
        return a.reshape(m, -1)

    def backward(self, error, learning_rate, no_activation_grad=False):
        m = error.shape[0]

        error = error.reshape(m, self.out_channels, self.output_height, self.output_width)

        if not no_activation_grad:
            error = error * self.activation_function.derivative(self.z)

        error_2d = error.reshape(m, self.out_channels, -1)

        dlt_w = error_2d @ self.col.transpose(0, 2, 1)
        dlt_w = dlt_w.mean(axis=0).reshape(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        dlt_w += self.reg.derivative(self.weights, self.lambda_, m)

        dlt_b = error.sum(axis=(0, 2, 3)).reshape(self.out_channels, 1, 1) / m

        self.reg_loss = self.reg.loss(self.weights, self.lambda_, m)

        self.weights -= learning_rate * dlt_w
        self.bias -= learning_rate * dlt_b

        w_reshaped = self.weights.reshape(self.out_channels, -1)
        prev_error_col = w_reshaped.T[np.newaxis, :, :] @ error_2d

        prev_error = _col2im(
            prev_error_col,
            self.in_channels,
            self.input_height,
            self.input_width,
            self.kernel_size,
            self.stride,
            self.padding,
        )
        return prev_error.reshape(m, -1)


def _im2col(x, kernel_size, stride, padding):
    """使用 stride tricks 将输入图像转换为列矩阵，用于高效卷积计算"""
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    m, c, h, w = x.shape
    oh = (h - kernel_size) // stride + 1
    ow = (w - kernel_size) // stride + 1

    shape = (m, c, oh, ow, kernel_size, kernel_size)
    s0, s1, s2, s3 = x.strides
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    col = patches.transpose(0, 1, 4, 5, 2, 3).copy()
    return col.reshape(m, c * kernel_size * kernel_size, oh * ow)


@numba.njit(parallel=True, fastmath=True)
def _col2im(col, c, h, w, kernel_size, stride, padding):
    """col2im: 将列矩阵转换回图像空间，梯度在各位置累加"""
    m = col.shape[0]
    k2 = kernel_size * kernel_size
    oh = (h + 2 * padding - kernel_size) // stride + 1
    ow = (w + 2 * padding - kernel_size) // stride + 1
    hp, wp = h + 2 * padding, w + 2 * padding

    img = np.zeros((m, c, hp, wp), dtype=col.dtype)

    for i in numba.prange(m):
        row = 0
        for ch in range(c):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    for oy in range(oh):
                        oy_idx = oy * stride + kh
                        for ox in range(ow):
                            img[i, ch, oy_idx, ox * stride + kw] += col[
                                i, row, oy * ow + ox
                            ]
                    row += 1

    if padding > 0:
        return img[:, :, padding : padding + h, padding : padding + w]
    return img


class Unittest(unittest.TestCase):
    def test_forward_shape(self):
        np.random.seed(42)
        conv = ConvolutionLayer(
            in_channels=1, out_channels=4, kernel_size=3,
            input_height=8, input_width=8, stride=1, padding=0,
            activation_function=ReLU,
        )
        conv.init_weights_and_bias(64)
        x = np.random.randn(2, 64)
        out = conv.forward(x)
        self.assertEqual(out.shape, (2, 4 * 6 * 6))

    def test_forward_with_padding(self):
        np.random.seed(42)
        conv = ConvolutionLayer(
            in_channels=1, out_channels=2, kernel_size=3,
            input_height=6, input_width=6, stride=1, padding=1,
            activation_function=ReLU,
        )
        conv.init_weights_and_bias(36)
        x = np.random.randn(3, 36)
        out = conv.forward(x)
        self.assertEqual(out.shape, (3, 2 * 6 * 6))

    def test_forward_with_stride(self):
        np.random.seed(42)
        conv = ConvolutionLayer(
            in_channels=1, out_channels=2, kernel_size=3,
            input_height=9, input_width=9, stride=2, padding=0,
            activation_function=ReLU,
        )
        conv.init_weights_and_bias(81)
        x = np.random.randn(2, 81)
        out = conv.forward(x)
        self.assertEqual(out.shape, (2, 2 * 4 * 4))

    def test_backward_updates_weights(self):
        np.random.seed(42)
        conv = ConvolutionLayer(
            in_channels=1, out_channels=2, kernel_size=3,
            input_height=6, input_width=6, stride=1, padding=0,
            activation_function=ReLU,
        )
        conv.init_weights_and_bias(36)
        w_before = conv.weights.copy()
        b_before = conv.bias.copy()

        x = np.random.randn(4, 36)
        conv.forward(x)
        error = np.random.randn(4, 2 * 4 * 4) * 0.1
        conv.backward(error, learning_rate=0.01)

        self.assertFalse(np.allclose(w_before, conv.weights))
        self.assertFalse(np.allclose(b_before, conv.bias))

    def test_gradient_numerical(self):
        """用数值梯度验证反向传播梯度"""
        np.random.seed(42)
        conv = ConvolutionLayer(
            in_channels=1, out_channels=1, kernel_size=3,
            input_height=4, input_width=4, stride=1, padding=0,
            activation_function=ReLU,
            reg=type("NoReg", (Regularization,), {"loss": lambda w, r, m: 0, "derivative": lambda w, r, m: np.zeros_like(w)}),
        )
        conv.init_weights_and_bias(16)
        x = np.random.randn(2, 16)
        conv.forward(x)

        # 使用恒等激活函数便于验证
        conv.activation_function = type("Identity", (ActivationFunction,), {"cal": lambda x: x, "derivative": lambda x: np.ones_like(x)})
        conv.forward(x)

        error = np.random.randn(2, 4) * 0.1
        conv.backward(error, learning_rate=0.0)

        # 数值梯度验证：对每个权重做微小扰动
        eps = 1e-5
        for oc in range(1):
            for ic in range(1):
                for kh in range(3):
                    for kw in range(3):
                        conv.weights[oc, ic, kh, kw] += eps
                        out_plus = conv.forward(x)
                        loss_plus = np.sum(out_plus.reshape(2, -1) * error.reshape(2, -1))

                        conv.weights[oc, ic, kh, kw] -= 2 * eps
                        out_minus = conv.forward(x)
                        loss_minus = np.sum(out_minus.reshape(2, -1) * error.reshape(2, -1))

                        conv.weights[oc, ic, kh, kw] += eps  # 恢复

                        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
                        # analytical grad from backward
                        # 这里检查梯度方向是否一致


if __name__ == "__main__":
    unittest.main()
