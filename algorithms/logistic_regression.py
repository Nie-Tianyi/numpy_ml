"""
logistic regression model
"""

import unittest
from typing import Optional, Tuple

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

from algorithms.activation_functions import sigmoid
from algorithms.loss_function import cross_entropy_loss
from algorithms.regularization import Regularization, lasso, ridge
from algorithms.standardizer import standardization
from test_data_set.test_data_gen import binary_data


class LogisticRegressionModel:
    """
    Logistic Regression Model
    """

    def __init__(
        self,
        niter: int = 1000,
        learning_rate: float = 0.01,
        reg_param: float = 0.3,
        regularization=Regularization.RIDGE,
    ):
        self.weights: Optional[NDArray[np.float64]] = None
        self.bias: Optional[NDArray[np.float64]] = None
        self.niter = niter
        self.lr = learning_rate
        self.lambda_ = reg_param
        self.regularization = regularization
        self.loss_history = []

    def __init_weights_and_bias(self, dim: int):
        self.weights = np.random.randn(dim)
        self.bias = np.zeros(1)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        :params x: 需要预测的数据， shape应该是(m,n)
        :returns: 返回模型预测值
        """
        assert x.shape[1] == self.weights.shape[0]

        return sigmoid(np.dot(x, self.weights) + self.bias)

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        训练模型
        :param x: 假设是一个 (m,n) shape的 numpy.ndarray，m表示有多少数据，n表示数据的维度
        :param y: labels，应该是一个 (m,1) shape的 numpy.ndarray
        """
        (m, n) = x.shape

        assert y.shape[0] == m, "x & y should have same length"

        self.__init_weights_and_bias(n)
        assert self.weights is not None and self.bias is not None, (
            "weights and bias should not be None"
        )

        for i in tqdm(range(self.niter)):
            y_hat = self.predict(x)

            if self.regularization != Regularization.NO_REGULARIZATION:
                loss = cross_entropy_loss(y_hat, y)
                self.loss_history.append(loss)

                (dlt_w, dlt_b) = self.__compute_gradient_without_regularization(
                    x, y_hat, y, m
                )
                self.weights -= self.lr * dlt_w
                self.bias -= self.lr * float(dlt_b)
            elif self.regularization != Regularization.LASSO:
                loss = cross_entropy_loss(y_hat, y) + lasso(
                    self.weights, self.lambda_, m
                )
                self.loss_history.append(loss)

                (dlt_w, dlt_b) = self.__compute_gradient_with_l1_regularization(
                    x, y_hat, y, m, self.lambda_, self.weights
                )
                self.weights -= self.lr * dlt_w
                self.bias -= self.lr * float(dlt_b)
            elif self.regularization != Regularization.RIDGE:
                loss = cross_entropy_loss(y_hat, y) + ridge(
                    self.weights, self.lambda_, m
                )
                self.loss_history.append(loss)

                (dlt_w, dlt_b) = self.__compute_gradient_with_l2_regularization(
                    x, y_hat, y, m, self.lambda_, self.weights
                )
                self.weights -= self.lr * dlt_w
                self.bias -= self.lr * float(dlt_b)

    @staticmethod
    def __compute_gradient_without_regularization(
        x: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        y_real: NDArray[np.float64],
        m: int,
    ) -> Tuple[NDArray[np.float64], np.float64]:
        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)
        return dlt_w, dlt_b

    @staticmethod
    def __compute_gradient_with_l1_regularization(
        x: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        y_real: NDArray[np.float64],
        m: int,
        lambda_: float,
        weights: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], np.float64]:
        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)

        dlt_w += np.sign(weights) * lambda_ / m

        return dlt_w, dlt_b

    @staticmethod
    def __compute_gradient_with_l2_regularization(
        x: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        y_real: NDArray[np.float64],
        m: int,
        lambda_: float,
        weights: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], np.float64]:
        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)

        dlt_w += weights * lambda_ / m

        return dlt_w, dlt_b

    def plot_loss_history(self, title="Training Loss History") -> None:
        """
        plot loss history
        """
        seaborn.lineplot(self.loss_history)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Cross-Entropy Loss")
        plt.show()


class Unittest(unittest.TestCase):
    def test_logistic_regression(self):
        (x, y) = binary_data(data_size=10000, seed=777)

        rescaled_x, scalar = standardization(x)

        model = LogisticRegressionModel(niter=500, learning_rate=0.1, reg_param=0.01)
        model.fit(rescaled_x, y)

        # 使用没有缩放过的数据训练的模型，作为对比
        model_no_scaled = LogisticRegressionModel(
            niter=500, learning_rate=1, reg_param=0.01
        )
        model_no_scaled.fit(x, y)

        # 处于 x_1 + x_2 = 1 右边的点预测结果应该大于0.5，并且离决策边际越远，预测结果越接近于1
        test_point = np.array([[1, 1]])
        res = model.predict(scalar.rescale(test_point))[0]
        print("\nFinal Results:")
        print(f"Predicted: {res:.4f}")  # 0.9958，有99.58%的概率这个点是1
        print(f"Weights: {model.weights}")
        print(f"Bias: {model.bias[0]:.4f}")

        model.plot_loss_history()
        model_no_scaled.plot_loss_history()
        # 使用缩放后的数据训练的模型收敛的又快又好
        print(f"Rescaled model's final loss: {model.loss_history[-1]:.4f}")
        print(f"Un-rescaled model's final loss: {model_no_scaled.loss_history[-1]:.4f}")

        self.assertAlmostEqual(res, 1, delta=0.1)


if __name__ == "__main__":
    unittest.main()
