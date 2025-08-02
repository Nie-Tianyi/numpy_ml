import unittest
import numpy as np
from numpy.typing import NDArray

from algorithms.regularization import Regularization


class PolynomialLogisticRegression:
    def __init__(
        self,
        niter=1000,
        learning_rate=0.01,
        reg_param=0.3,
        regularization=Regularization.RIDGE,
    ):
        self.weights = None
        self.bias = None
        self.niter = niter
        self.lr = learning_rate
        self.lambda_ = reg_param
        self.reg = regularization
        self.loss_history = []

    def __init_weights_and_bias(self, dim: int, k: int):
        # dim 数据有多少个维度；k 多分类问题里面有多少个预测分类
        self.weights = np.random.rand(dim, k)
        self.bias = np.zeros(k)

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


class Unittest(unittest.TestCase):
    def test_softmax(self):
        pass


if __name__ == "__main__":
    unittest.main()
