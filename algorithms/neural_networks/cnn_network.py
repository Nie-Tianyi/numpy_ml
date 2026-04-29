"""
CNN 卷积神经网络，用于图像分类
"""

import unittest
from typing import List, Optional

import numpy as np

from algorithms.activation_functions import ReLU, Softmax
from algorithms.evaluation import EvaluationMethod, Accuracy
from algorithms.loss_function import sparse_categorical_cross_entropy_loss
from algorithms.neural_networks.convolution_layer import ConvolutionLayer
from algorithms.neural_networks.linear_layer import FCLinearLayer
from algorithms.neural_networks.neural_network import NeuralNetworkBaseModel
from algorithms.neural_networks.neural_network_layer_abstract import NeuralNetworkLayerAbstract
from algorithms.neural_networks.pooling_layer import MaxPoolingLayer
from algorithms.normaliser import max_min_normalisation
from algorithms.polynomial_logistic_regression import PolynomialLogisticRegression
from algorithms.regularization import Regularization, Ridge
from test_data_set.linear_data import binary_data
from test_data_set.mnist import mnist


class CNNNetwork(NeuralNetworkBaseModel):
    """
    CNN 卷积神经网络模型，用于多分类任务
    默认结构：Conv -> Pool -> Conv -> Pool -> FC -> Softmax
    """

    def __init__(
        self,
        k: int,
        layers: Optional[List[NeuralNetworkLayerAbstract]] = None,
        niter=1000,
        learning_rate=0.1,
        reg_param=0.01,
        regularization: type[Regularization] = Ridge,
    ):
        if layers is None:
            layers = [
                ConvolutionLayer(
                    in_channels=1, out_channels=8, kernel_size=3,
                    input_height=28, input_width=28, stride=1, padding=1,
                ),
                MaxPoolingLayer(input_channels=8, input_height=28, input_width=28, pool_size=2, stride=2),
                ConvolutionLayer(
                    in_channels=8, out_channels=16, kernel_size=3,
                    input_height=14, input_width=14, stride=1, padding=1,
                ),
                MaxPoolingLayer(input_channels=16, input_height=14, input_width=14, pool_size=2, stride=2),
                FCLinearLayer(128, activation_function=ReLU),
                FCLinearLayer(k, activation_function=Softmax),
            ]

        super().__init__(
            layers,
            sparse_categorical_cross_entropy_loss,
            niter=niter,
            learning_rate=learning_rate,
            reg_param=reg_param,
            regularization=regularization,
        )

        self.labels = None

    def fit(self, x, y):
        self.labels = np.unique(y)
        y = (y == self.labels).astype(np.float64)
        super().fit(x, y)

    def predict_label(self, x):
        poss = self.predict(x)
        return self.labels[np.argmax(poss, axis=1)].reshape(-1, 1)

    def evaluate(
        self, x_test, y_test, evaluation_method: type[EvaluationMethod] = Accuracy
    ) -> float:
        y_hat = self.predict_label(x_test)
        return evaluation_method.evaluate(y_hat, y_test)


class Unittest(unittest.TestCase):
    def test_binary(self):
        (x, y) = binary_data(data_size=200, seed=1)
        x, scaler = max_min_normalisation(x)

        # 2D 数据作为 1 通道 1x2 "图像"
        cnn = CNNNetwork(
            k=2,
            layers=[
                ConvolutionLayer(
                    in_channels=1, out_channels=4, kernel_size=1,
                    input_height=1, input_width=2, stride=1, padding=0,
                ),
                FCLinearLayer(4, activation_function=ReLU),
                FCLinearLayer(2, activation_function=Softmax),
            ],
            niter=500,
            learning_rate=0.01,
        )
        pl_model = PolynomialLogisticRegression()

        cnn.fit(x, y)
        pl_model.fit(x, y)

        (test_x, test_y) = binary_data(data_size=100, seed=2)
        test_x = scaler.rescale(test_x)
        acc_cnn = cnn.evaluate(test_x, test_y)
        acc_plr = pl_model.evaluate(test_x, test_y)
        print("CNN's Accuracy:", acc_cnn)
        print("Polynomial Logistic Regression Model's Accuracy:", acc_plr)

        self.assertAlmostEqual(acc_cnn, acc_plr, delta=0.15)

    def test_mnist_small(self):
        """用小批量 MNIST 验证 CNN 能够训练并降低损失"""
        (x, y) = mnist(data_size=2000, seed=1)

        def reshape(arr):
            return arr.reshape(arr.shape[0], -1)

        x = reshape(x)
        x, scaler = max_min_normalisation(x)
        x_train, x_test = x[:1500], x[1500:]
        y_train, y_test = y[:1500], y[1500:]

        cnn = CNNNetwork(
            k=10,
            layers=[
                ConvolutionLayer(
                    in_channels=1, out_channels=4, kernel_size=3,
                    input_height=28, input_width=28, stride=1, padding=1,
                ),
                MaxPoolingLayer(input_channels=4, input_height=28, input_width=28, pool_size=2, stride=2),
                ConvolutionLayer(
                    in_channels=4, out_channels=8, kernel_size=3,
                    input_height=14, input_width=14, stride=1, padding=1,
                ),
                MaxPoolingLayer(input_channels=8, input_height=14, input_width=14, pool_size=2, stride=2),
                FCLinearLayer(32, activation_function=ReLU),
                FCLinearLayer(10, activation_function=Softmax),
            ],
            niter=200,
            learning_rate=0.01,
        )
        cnn.fit(x_train, y_train)
        print("Model's Final Loss:", cnn.loss_history[-1])
        print("Model's Initial Loss:", cnn.loss_history[0])

        acc = cnn.evaluate(x_test, y_test)
        print("Accuracy:", acc)

        self.assertLess(cnn.loss_history[-1], cnn.loss_history[0] * 0.8)
        self.assertGreater(acc, 0.30)


if __name__ == "__main__":
    unittest.main()
