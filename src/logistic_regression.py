import unittest

import numba
import numpy as np

from src.regularization import RegularizationTerm


class LogisticRegressionModel:
    """
    Logistic Regression Model
    """

    def __init__(self, niter: int = 1000, learning_rate: float = 0.01, reg_param: float = 0.3,
                 regularization=RegularizationTerm.RIDGE):
        self.weights = None
        self.bias = None
        self.niter = niter
        self.lr = learning_rate
        self.lambda_ = reg_param
        self.regularization = regularization
        self.loss_history = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        训练模型
        :param x: 假设是一个 (m,n) shape的 numpy.ndarray，m表示有多少数据，n表示数据的维度
        :param y: labels，应该是一个 (m,1) shape的 numpy.ndarray
        """
        (m, n) = x.shape

        assert y.shape[0] == m, "x & y should have same length"

        self.__init_weights_and_bias(n)

        pass

    def predict(self, x: np.ndarray):
        """
        :params x: 需要预测的数据， shape应该是(m,n)
        :returns: 返回模型预测值
        """
        assert x.shape[1] == self.weights.shape[0]

        return self.__predict(self.weights, self.bias, x)

    @staticmethod
    @numba.njit(fastmath=True)
    def __predict(weights, bias, x):
        return sigmoid(np.dot(x, weights) + bias)

    def __init_weights_and_bias(self, dim: int):
        self.weights = np.random.randn(dim)
        self.bias = np.zeros(1)


@numba.njit(fastmath=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Unittest(unittest.TestCase):
    def test_logistic_regression(self):
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
