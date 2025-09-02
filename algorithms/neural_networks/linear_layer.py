"""
线性层神经网络
"""

import numpy as np
import numba

from algorithms.activation_functions import ActivationFunction, ReLU
from algorithms.neural_networks.neural_network_layer_abstract import NeuralNetworkLayerAbstract
from algorithms.regularization import Regularization, Ridge


class FCLinearLayer(NeuralNetworkLayerAbstract):
    """
    Fully Connected Linear Layer
    全连接线性层，默认使用ReLU作为激活函数
    """

    def __init__(
        self,
        num,
        activation_function: type[ActivationFunction] = ReLU,
        reg: type[Regularization] = Ridge,
        reg_params=0.1,
    ):
        super().__init__(num)
        self.activation_function = activation_function  # 激活函数，默认是ReLU
        self.reg = reg  # 正则化，默认是L2正则
        self.lambda_ = reg_params  # 正则化超参数，默认0.1
        self.z = None  # 线性输出，形状为 (m, num)，m是输入数据的长度
        self.inputs = None  # 上一层的激活输出，这一层的输入，形状为 (m, num)

    def init_weights_and_bias(self, dim):
        self.weights = np.random.randn(self.num, dim) * np.sqrt(2.0 / dim)  # 使用He初始化权重
        self.bias = np.zeros(self.num)

    def forward(self, x):
        """
        前向传播
        :param x: 输入，形状为 (m, dim)
        :return: 输出，形状为 (m, num)
        """
        self.inputs = x
        # weights.shape = (num, dim) x.shape = (m, dim)
        self.z = np.dot(x, self.weights.T) + self.bias
        # z.shape = (m, num)
        a = self.activation_function.cal(self.z)
        return a

    def backward(self, error, learning_rate, no_activation_grad=False):
        """
        反向传播：更新自己的权重，然后返回下一层的误差 delta
        :param error: 这一层的误差，**不包括这一层激活函数的梯度**，形状为 (m, num)
        :return: 下一层的误差，**同样也不包括下一层的激活函数的梯度**，形状为 (m, dim)
        :param no_activation_grad: 是否计算激活函数的梯度，默认计算。
        :param learning_rate: 学习率
        """
        m = error.shape[0]
        # 先把激活函数的梯度算上
        # print("before activation gradient:", error) # 梯度爆炸
        if not no_activation_grad:
            error = error * self.activation_function.derivative(self.z)  # error.shape = (m, num)
        # print("after:", error)

        # 计算下一层的error（要在更新参数之前）
        prev_layer_error = np.dot(error, self.weights)
        self.reg_loss = self.reg.loss(self.weights, self.lambda_, m)

        # 计算梯度 更新参数
        dlt_w, dlt_b = _compute_gradient(error, self.inputs, m)
        # 加上正则化带来的梯度
        dlt_w += self.reg.derivative(self.weights, self.lambda_, m)

        # 更新参数
        self.weights -= learning_rate * dlt_w
        self.bias -= learning_rate * dlt_b

        return prev_layer_error


# 最关键的计算部分用numba加速
@numba.njit(parallel=True, fastmath=True)
def _compute_gradient(error, inputs, m):
    dlt_w = (1 / m) * np.dot(error.T, inputs)  # dlt_w.shape = (num, dim)
    dlt_b = (1 / m) * np.sum(error, axis=0)  # dlt_b.shape = (num,)
    return dlt_w, dlt_b
