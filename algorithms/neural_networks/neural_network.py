"""
神经网络基础模型
"""

from abc import ABC
from typing import List

from tqdm import tqdm

from algorithms.model_abstract import MachineLearningModel
from algorithms.neural_networks.neural_network_layer_abstract import NeuralNetworkLayerAbstract
from algorithms.regularization import Regularization, Ridge


class NeuralNetworkBaseModel(MachineLearningModel, ABC):
    """
    神经网络基础模型，任何其他模型继承自这个基础模型
    继承这个模型至少需要自己实现 __init__() 以及 evaluate()
    """

    def __init__(
        self,
        layers: List[NeuralNetworkLayerAbstract],
        loss_function,
        niter=1000,
        learning_rate=0.1,
        reg_param: float = 0.3,
        regularization: type[Regularization] = Ridge,
    ):
        super().__init__(regularization, niter, learning_rate, reg_param)
        self.layers = layers
        self.loss_function = loss_function

    def __init_weights_and_bias(self, dim):
        # 逐层初始化权重和bias
        for layer in self.layers:
            layer.init_weights_and_bias(dim)
            dim = layer.num

    def fit(self, x, y):
        # 先初始化神经网络的参数
        (m, dim) = x.shape
        self.__init_weights_and_bias(dim)
        for _ in tqdm(range(self.niter)):
            y_hat = self.forward_propagation(x)
            reg_loss = self.backward_propagation(y_hat - y)
            self.loss_history.append(self.loss_function(y_hat, y) + reg_loss)

    def predict(self, x):
        return self.forward_propagation(x)

    def forward_propagation(self, x):
        """
        前向传播
        :param x: 输入数据
        :return: 预测结果
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward_propagation(self, error) -> float:
        """
        反向传播，更新模型参数
        :param error: y_hat - y
        :return: 返回每一层正则化的损失
        """
        reg_loss = 0  # 记录每一层的正则化项带来的损失
        for layer in reversed(self.layers):
            # 最后一层神经网络不计算激活函数的梯度，因为error里面已经包含了
            if layer is self.layers[-1]:
                error = layer.backward(error, learning_rate=self.lr, no_activation_grad=True)
            else:
                error = layer.backward(error, learning_rate=self.lr)
            reg_loss += layer.reg_loss
        return reg_loss

    @property
    def weights(self):
        """
        每一层神经网络权重
        :return: 返回每一层神经网络权重
        """
        return [layer.weights for layer in self.layers]

    @property
    def biases(self):
        """
        :return: 返回每一层神经网络偏置
        """
        return [layer.bias for layer in self.layers]
