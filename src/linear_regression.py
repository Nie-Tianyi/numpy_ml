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

    def __init__(self, niter=1000, learning_rate=0.1, regula_param=0):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.lambda_ = regula_param
        self.niter = niter
        self.loss_history = []

    def fit(self, x, y, regularization=RegularizationTerm.RIDGE):
        assert x.shape[1] == y, "x and y must be the same length"
        (dim, m) = x.shape
        self._init_weights_and_bias(dim)

        for _ in range(self.niter):
            y_hat = self.predict(x)

            loss = mean_square_error(y_hat, y)
            self.loss_history.append(loss)
            # update weights
            self._compute_gradient(x, y_hat, y)
            if regularization == RegularizationTerm.RIDGE:
                self.weights -= self.lr * (self.lambda_ / m) * self.weights

    def predict(self, x):
        return np.dot(x, self.weights.T) + self.bias

    def _init_weights_and_bias(self, dim):
        self.weights = np.zeros(dim)
        self.bias = np.zeros(1)

    def _compute_gradient(self, x, y_pred, y_real):
        dlt_w = np.mean((y_pred - y_real) * x.T, axis=0)
        dlt_b = np.mean(y_pred - y_real)

        self.weights -= dlt_w * self.lr
        self.bias -= dlt_b * self.lr


class Unittest(unittest.TestCase):
    def test_model(self):
        pass


if __name__ == "__main__":
    unittest.main()
