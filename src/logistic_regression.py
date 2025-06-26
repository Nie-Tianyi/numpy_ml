import unittest

import numpy as np


class LogisticRegressionModel:
    def __init__(self, niter: int = 1000, learning_rate: float = 0.01, reg_param: float = 0.3):
        self.weights = None
        self.bias = None
        self.niter = niter
        self.lr = learning_rate
        self.lambda_ = reg_param
        self.loss_history = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

class Unittest(unittest.TestCase):
    def test_logistic_regression(self):
        self.assertEqual(1 + 1, 2)
