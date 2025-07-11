"""
logistic regression model
"""
import unittest

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from src.activision_functions import sigmoid
from src.loss_function import cross_entropy_loss
from src.regularization import RegularizationTerm
from src.standardizer import standardization
from test_data_set.test_data_gen import binary_data


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

    def __init_weights_and_bias(self, dim: int):
        self.weights = np.random.randn(dim)
        self.bias = np.zeros(1)

    def predict(self, x: np.ndarray):
        """
        :params x: 需要预测的数据， shape应该是(m,n)
        :returns: 返回模型预测值
        """
        assert x.shape[1] == self.weights.shape[0]

        return sigmoid(np.dot(x, self.weights) + self.bias)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        训练模型
        :param x: 假设是一个 (m,n) shape的 numpy.ndarray，m表示有多少数据，n表示数据的维度
        :param y: labels，应该是一个 (m,1) shape的 numpy.ndarray
        """
        (m, n) = x.shape

        assert y.shape[0] == m, "x & y should have same length"

        self.__init_weights_and_bias(n)
        assert self.weights is not None and self.bias is not None, "weights and bias should not be None"

        for i in range(self.niter):
            y_hat = self.predict(x)
            loss = cross_entropy_loss(y_hat, y)
            self.loss_history.append(loss)

            if self.regularization != RegularizationTerm.No_REGULARIZATION:
                (dlt_w, dlt_b) = self.__compute_gradient_without_regularization(x, y_hat, y, m)
                self.weights -= self.lr * dlt_w
                self.bias -= self.lr * float(dlt_b)
            elif self.regularization != RegularizationTerm.LASSO:
                (dlt_w, dlt_b) = self.__compute_gradient_with_l1_regularization(x, y_hat, y, m, self.lambda_,
                                                                                self.weights)
                self.weights -= self.lr * dlt_w
                self.bias -= self.lr * float(dlt_b)
            elif self.regularization != RegularizationTerm.RIDGE:
                (dlt_w, dlt_b) = self.__compute_gradient_with_l2_regularization(x, y_hat, y, m, self.lambda_,
                                                                                self.weights)
                self.weights -= self.lr * dlt_w
                self.bias -= self.lr * float(dlt_b)

    @staticmethod
    def __compute_gradient_without_regularization(x: np.ndarray, y_pred: np.ndarray, y_real: np.ndarray, m: int):
        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)
        return dlt_w, dlt_b

    @staticmethod
    def __compute_gradient_with_l1_regularization(x: np.ndarray, y_pred: np.ndarray, y_real: np.ndarray, m: int,
                                                  lambda_: float, weights: np.ndarray):
        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)

        dlt_w += np.sign(weights) * lambda_ / m

        return dlt_w, dlt_b

    @staticmethod
    def __compute_gradient_with_l2_regularization(x: np.ndarray, y_pred: np.ndarray, y_real: np.ndarray, m: int,
                                                  lambda_: float, weights: np.ndarray):
        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)

        dlt_w += weights * lambda_ / m

        return dlt_w, dlt_b

    def plot_loss_history(self):
        """
        plot loss history
        """
        seaborn.lineplot(self.loss_history)
        plt.title("Training Loss History")
        plt.xlabel("Iteration")
        plt.ylabel("Cross-Entropy Loss")
        plt.show()


class Unittest(unittest.TestCase):
    def test_logistic_regression(self):
        (x, y) = binary_data(data_size=10000, seed=777)

        rescaled_x, scalar = standardization(x)

        model = LogisticRegressionModel(niter=500, learning_rate=0.1, reg_param=0.01)
        model.fit(rescaled_x, y)

        model_no_scaled = LogisticRegressionModel(niter=500, learning_rate=1, reg_param=0.01)
        model_no_scaled.fit(x, y)

        test_point = np.array([[1, 1]])
        res = model.predict(scalar.rescale(test_point))[0]
        print("\nFinal Results:")
        print(f"Predicted: {res:.4f}")
        print(f"Weights: {model.weights}")
        print(f"Bias: {model.bias[0]:.4f}")

        model.plot_loss_history()
        model_no_scaled.plot_loss_history()
        print(f"Rescaled model's final loss: {model.loss_history[-1]:.4f}")
        print(f"Un-rescaled model's final loss: {model_no_scaled.loss_history[-1]:.4f}")

        self.assertAlmostEqual(res, 1, delta=0.1)


if __name__ == "__main__":
    unittest.main()
