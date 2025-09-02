"""
多分类神经网络，可以自定义神经网络结构，单输出层必须是个数为K，而且激活函数为Softmax的神经元
"""

import unittest

from typing import List, Optional

import numpy as np

from algorithms.activation_functions import Softmax, ReLU, LeakyReLU
from algorithms.evaluation import EvaluationMethod, Accuracy
from algorithms.loss_function import sparse_categorical_cross_entropy_loss
from algorithms.neural_networks.linear_layer import FCLinearLayer
from algorithms.neural_networks.neural_network import NeuralNetworkBaseModel
from algorithms.neural_networks.neural_network_layer_abstract import NeuralNetworkLayerAbstract
from algorithms.normaliser import max_min_normalisation, z_score_normalisation
from algorithms.polynomial_logistic_regression import PolynomialLogisticRegression
from algorithms.regularization import Regularization, Ridge
from test_data_set.mnist import mnist
from test_data_set.linear_data import binary_data


class MultiClassificationNeuralNetwork(NeuralNetworkBaseModel):
    def evaluate(
        self, x_test, y_test, evaluation_method: type[EvaluationMethod] = Accuracy
    ) -> float:
        y_hat = self.predict_label(x_test)
        return evaluation_method.evaluate(y_hat, y_test)

    def __init__(
        self,
        k: int,
        layers: Optional[List[NeuralNetworkLayerAbstract]] = None,
        niter=1000,
        learning_rate=0.1,
        reg_param=0.01,
        regularization: type[Regularization] = Ridge,
    ):
        """
        多分类神经网络
        :param k: 标签类别的个数
        :param layers: 默认是 4，2，k，激活函数分别是 ReLU ReLU Softmax
        :param niter: 循环多少次
        :param learning_rate: 学习率
        :param reg_param: 正则化超参数
        :param regularization: 正则化项，默认使用L2正则化
        """
        if layers is None:
            layers = [
                FCLinearLayer(4, activation_function=ReLU),
                FCLinearLayer(2, activation_function=ReLU),
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
        print("possible labels:", self.labels)
        # 将 y 转换成one-hot编码
        y = (y == self.labels).astype(np.float64)
        super().fit(x, y)

    def predict_label(self, x):
        poss = self.predict(x)
        return self.labels[np.argmax(poss, axis=1)].reshape(-1, 1)


class Unittest(unittest.TestCase):
    def test_binary(self):
        (x, y) = binary_data(data_size=1000, seed=1)
        x, scaler = z_score_normalisation(x)

        neural_network = MultiClassificationNeuralNetwork(2)
        pl_model = PolynomialLogisticRegression()

        neural_network.fit(x, y)
        neural_network.plot_loss_history(title="Neural Network's Loss History")
        pl_model.fit(x, y)
        pl_model.plot_loss_history(title="Polynomial Logistic Regression's Loss History")

        (test_x, test_y) = binary_data(data_size=100, seed=2)
        test_x = scaler.rescale(test_x)
        acc_nn = neural_network.evaluate(test_x, test_y)
        acc_plr = pl_model.evaluate(test_x, test_y)
        print("Neural Network's Accuracy:", acc_nn)
        print("Polynomial Logistic Regression Model's Accuracy:", acc_plr)

        self.assertAlmostEqual(acc_nn, acc_plr, delta=0.1)

    def test_multiclass_nn(self):
        (x, y) = mnist(data_size=70000, seed=1)

        def reshape(arr):
            return arr.reshape(arr.shape[0], -1)

        x = reshape(x)
        x, scaler = max_min_normalisation(x)
        x_train, x_test = x[:60000], x[60000:]
        y_train, y_test = y[:60000], y[60000:]

        neural_network = MultiClassificationNeuralNetwork(
            k=10,
            layers=[
                FCLinearLayer(256, activation_function=LeakyReLU),
                FCLinearLayer(128, activation_function=LeakyReLU),
                FCLinearLayer(64, activation_function=LeakyReLU),
                FCLinearLayer(10, activation_function=Softmax),
            ],
            niter=1000,
            learning_rate=0.01,
        )
        neural_network.fit(x_train, y_train)
        neural_network.plot_loss_history(title="MNIST", label="Sparse Cross Entropy Loss")
        print("Model's Final Loss:", neural_network.loss_history[-1])

        acc = neural_network.evaluate(x_test, y_test)
        print("Accuracy:", acc)

        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
