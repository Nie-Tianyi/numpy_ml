"""
Linear Regression Model
"""
import unittest

import numpy as np

from src.loss_function import mean_square_error
from src.regularization import RegularizationTerm


class LinearRegressionModel:
    """
    Linear Regression Model
    """

    def __init__(self, niter=1000, learning_rate=0.1, regula_param=0.3):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.lambda_ = regula_param
        self.niter = niter
        self.loss_history = []

    def fit(self, x, y, regularization=RegularizationTerm.RIDGE):
        assert x.shape[0] == y.shape[0], "x and y must be the same length"
        (m, dim) = x.shape
        y = y.flatten()
        self._init_weights_and_bias(dim)

        for _ in range(self.niter):
            y_hat = self.predict(x)
            loss = mean_square_error(y_hat, y)
            self.loss_history.append(loss)
            # update weights
            self._compute_gradient(x, y_hat, y)
            if regularization == RegularizationTerm.RIDGE:
                self.weights -= self.lr * (self.lambda_ / m) * self.weights

        print(self.loss_history)

    def predict(self, x):
        return np.dot(x, self.weights) + self.bias

    def _init_weights_and_bias(self, dim):
        self.weights = np.zeros(dim)
        self.bias = np.zeros(1)

    def _compute_gradient(self, x, y_pred, y_real):
        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / len(y_real)
        dlt_b = np.mean(error)

        self.weights -= self.lr * dlt_w
        self.bias -= self.lr * dlt_b

        print(self.weights, self.bias)


class Unittest(unittest.TestCase):
    def test_linear_model(self):
        data_size = 10000
        np.random.seed(7)
        x_1 = np.linspace(0, 1000, num=data_size)
        x_2 = np.linspace(20, 999, num=data_size)
        noise = np.random.randn(data_size)  # 创建1维噪声数组

        # 创建特征矩阵 (10000, 2)
        x = np.vstack([x_1, x_2]).T

        # 创建目标值 (10000,)
        y = 0.99 * x_1 + 2.3 * x_2 + noise + 1

        model = LinearRegressionModel(niter=5000, learning_rate=0.01)
        model.fit(x, y)

        # 预测时需要二维输入 [[1, 1]]
        res = model.predict(np.array([[1, 1]]))[0]

        # 允许数值误差
        self.assertAlmostEqual(res, 4.29, delta=0.1)


if __name__ == "__main__":
    unittest.main()
