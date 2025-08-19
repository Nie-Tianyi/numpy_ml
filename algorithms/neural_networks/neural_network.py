from abc import ABC
from typing import List

from tqdm import tqdm

from algorithms.model_abstract import MachineLearningModel
from algorithms.neural_networks.neural_network_layer_abstract import NeuralNetworkLayer
from algorithms.regularization import Regularization, Ridge


class NeuralNetwork(MachineLearningModel, ABC):
    """
    neural network base model, any other neural network inherit from this base model
    """

    def __init__(
        self,
        layers: List[NeuralNetworkLayer],
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
        y = y.reshape(-1, 1)
        for _ in tqdm(range(self.niter)):
            y_hat = self.forward_propagation(x)
            reg_loss = self.backward_propagation(y_hat - y)
            self.loss_history.append(self.loss_function(y_hat, y) + reg_loss)

    def predict(self, x):
        return self.forward_propagation(x)

    def forward_propagation(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward_propagation(self, error) -> float:
        reg_loss = 0  # 记录每一层的正则化项带来的损失
        for layer in reversed(self.layers):
            if layer is self.layers[-1]:
                error = layer.backward(error, no_activation=True)
            else:
                error = layer.backward(error)
            reg_loss += layer.reg_loss
        return reg_loss
