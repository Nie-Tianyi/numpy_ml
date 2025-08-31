"""
神经网络层抽象
"""

from abc import ABC, abstractmethod


# 定义接口
class NeuralNetworkLayerAbstract(ABC):
    """
    神经网络单层抽象接口
    """

    def __init__(self, num):
        self.num = num
        self.reg_loss = 0
        self.weights = None
        self.bias = None

    @abstractmethod
    def init_weights_and_bias(self, dim):
        """初始化这一层的参数"""
        pass

    @abstractmethod
    def forward(self, x):
        """前向传播"""
        pass

    @abstractmethod
    def backward(self, error, learning_rate, no_activation_grad=False):
        """反向传播，跟新参数"""
        pass
