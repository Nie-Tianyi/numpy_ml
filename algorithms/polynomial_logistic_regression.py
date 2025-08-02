import unittest

from algorithms.regularization import Regularization
from test_data_set.mnist import mnist


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
        pass

    def fit(self):
        pass

    def predict(self, x):
        pass


class Unittest(unittest.TestCase):
    def test_mnist(self):
        data = mnist()
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
