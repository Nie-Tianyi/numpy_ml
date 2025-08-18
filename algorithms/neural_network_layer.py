from abc import ABC, abstractmethod

import numpy as np

from algorithms.activation_functions import ActivationFunction, ReLU, Sigmoid
from algorithms.regularization import Regularization, Ridge


# 定义trait
class NeuralNetworkLayer(ABC):
    def __init__(self, num):
        self.num = num
        self.reg_loss = 0

    @abstractmethod
    def init_weights_and_bias(self, dim):
        """初始化这一层的参数"""
        pass

    @abstractmethod
    def forward(self, x):
        """前向传播"""
        pass

    @abstractmethod
    def backward(self, error):
        """反向传播，跟新参数"""
        pass


class LinearLayer(NeuralNetworkLayer):
    """
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
        self.weights = None  # 权重，形状为 (num, dim)
        self.bias = None  # 偏置，形状为 (num,)
        self.z = None  # 线性输出，形状为 (m, num)，m是输入数据的长度
        self.inputs = None  # 上一层的激活输出，这一层的输入，形状为 (m, num)

    def init_weights_and_bias(self, dim):
        self.weights = np.random.randn(self.num, dim)
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

    def backward(self, error):
        """
        反向传播：更新自己的权重，然后返回下一层的误差 delta
        :param error: 这一层的误差，**不包括这一层激活函数的梯度**，形状为 (m, num)
        :return: 下一层的误差，**同样也不包括下一层的激活函数的梯度**，形状为 (m, dim)
        """
        m = error.shape[0]
        # 先把激活函数的梯度算上
        error = error * self.activation_function.derivative(self.z)  # error.shape = (m, num)

        # 计算下一层的error（要在更新参数之前）
        prev_layer_error = np.dot(error, self.weights)
        self.reg_loss = self.reg.loss(self.weights, self.lambda_, m)

        # 计算梯度 更新参数
        dlt_w = (1 / m) * np.dot(error.T, self.inputs)  # dlt_w.shape = (num, dim)
        dlt_b = (1 / m) * np.sum(error)  # dlt_b.shape = (num,)
        # 加上正则化带来的梯度
        dlt_w += self.reg.derivative(self.weights, self.lambda_, m)
        # 更新参数
        self.weights -= dlt_w
        self.bias -= dlt_b

        return prev_layer_error


class SigmoidOutputLayer(LinearLayer):
    def __init__(
        self,
        reg: type[Regularization] = Ridge,
        reg_params=0.1,
    ):
        super().__init__(1, Sigmoid, reg, reg_params)

    def init_weights_and_bias(self, dim):
        super().init_weights_and_bias(dim)

    def forward(self, x):
        return super().forward(x)

    def backward(self, error):
        """
        反向传播：更新自己的权重，然后返回下一层的误差 delta
        :param error: y_hat - y
        :return: 下一层的误差，**不包括下一层的激活函数的梯度**，形状为 (m, dim)
        """
        m = error.shape[0]
        # 计算下一层的error（要在更新参数之前）
        prev_layer_error = np.dot(error, self.weights)

        # 计算梯度 更新参数
        dlt_w = (1 / m) * np.dot(error.T, self.inputs)  # dlt_w.shape = (num, dim)
        dlt_b = (1 / m) * np.sum(error)  # dlt_b.shape = (num,)
        # 加上正则化带来的梯度
        dlt_w += self.reg.derivative(self.weights, self.lambda_, m)
        # 更新参数
        self.weights -= dlt_w
        self.bias -= dlt_b

        return prev_layer_error
