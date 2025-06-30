"""
Fixed Linear Regression Model
"""
import unittest

import numba
import numpy as np
import seaborn
from matplotlib import pyplot as plt

from src.loss_function import mean_square_error
from src.regularization import RegularizationTerm
from src.standardizer import standardization
from test_data_set.test_data_gen import linear_data


class LinearRegressionModel:
    """
    Linear Regression Model
    """

    def __init__(self, niter=1000, learning_rate=0.01, reg_param=0.3, regularization=RegularizationTerm.RIDGE):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.lambda_ = reg_param
        self.niter = niter
        self.regularization = regularization
        self.loss_history = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        训练模型，x对应着数据，y对应着label，regularization代表正则化方式
        :param x: 假设是一个 (m,n) shape的 numpy.ndarray，m表示有多少数据，n表示数据的维度
        :param y: labels，应该是一个 (m,1) shape的 numpy.ndarray
        """
        assert x.shape[0] == y.shape[0], "x and y must be the same length"
        (m, dim) = x.shape
        y = y.flatten()

        self.__init_weights_and_bias(dim)
        assert self.weights is not None and self.bias is not None, "Weights and bias must be initialized"

        for i in range(self.niter):
            y_hat = self.predict(x)
            loss = mean_square_error(y_hat, y)
            self.loss_history.append(loss)

            # 更新梯度 (包含正则化)
            if self.regularization == RegularizationTerm.No_REGULARIZATION:
                (dlt_w, dlt_b) = self.__compute_gradient_without_regularization(x, y_hat, y, m)
                self.weights -= self.lr * dlt_w
                self.bias -= self.lr * dlt_b
            elif self.regularization == RegularizationTerm.LASSO:
                (dlt_w, dlt_b) = self.__computer_gradient_with_l1_regularization(x, y_hat, y, m, self.lambda_,
                                                                                 self.weights)
                self.weights -= self.lr * dlt_w
                self.bias -= self.lr * dlt_b
            elif self.regularization == RegularizationTerm.RIDGE:
                (dlt_w, dlt_b) = self.__computer_gradient_with_l2_regularization(x, y_hat, y, m, self.lambda_,
                                                                                 self.weights)
                self.weights -= self.lr * dlt_w
                self.bias -= self.lr * dlt_b

            # 每100次迭代打印进度
            if i % 10 == 0:
                print(f"Iteration {i}: Loss={loss:.4f}, Weights={self.weights}, Bias={self.bias[0]:.4f}")

    def predict(self, x: np.ndarray):
        """
        预测数据
        :param x:  should be the same length as weights
        :return: float predicted value
        """
        if self.weights is None:
            raise ValueError("Model has not been initialised yet")
        assert x.shape[1] == self.weights.shape[0]
        return np.dot(x, self.weights) + self.bias

    def __init_weights_and_bias(self, dim: int):
        # 初始化权重和偏置
        self.weights = np.random.rand(dim)
        self.bias = np.zeros(1)

    @staticmethod
    @numba.jit(fastmath=True)
    def __compute_gradient_without_regularization(x: np.ndarray, y_pred: np.ndarray, y_real: np.ndarray, m: int) -> \
            tuple[np.ndarray, np.floating]:
        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)

        return dlt_w, dlt_b

    @staticmethod
    @numba.njit(fastmath=True)
    def __computer_gradient_with_l2_regularization(x: np.ndarray, y_pred: np.ndarray, y_real: np.ndarray, m: int,
                                                   lambda_: float, weights: np.ndarray) -> tuple[
        np.ndarray, np.floating]:

        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)
        dlt_w += (lambda_ / m) * weights

        return dlt_w, dlt_b

    @staticmethod
    @numba.njit(fastmath=True)
    def __computer_gradient_with_l1_regularization(x: np.ndarray, y_pred: np.ndarray, y_real: np.ndarray, m: int,
                                                   lambda_: float, weights: np.ndarray) -> tuple[
        np.ndarray, np.floating]:

        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)
        dlt_w += (lambda_ / m) * np.sign(weights)

        return dlt_w, dlt_b

    def plot_loss_history(self):
        """
        plot loss history
        """
        seaborn.lineplot(self.loss_history)
        plt.title("Training Loss History")
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.show()


class Unittest(unittest.TestCase):

    def test_linear_model(self):
        x, y = linear_data(data_size=10000, seed=777)

        model = LinearRegressionModel(niter=100, learning_rate=0.1, reg_param=0.1)
        model.fit(x, y)

        # 测试点需要是2D数组
        test_point = np.array([[1, 1]])
        res = model.predict(test_point)[0]

        print("\nFinal Results:")
        print(f"Predicted: {res:.4f}")
        print(f"Weights: {model.weights}")
        print(f"Bias: {model.bias[0]:.4f}")

        # 绘制损失曲线
        model.plot_loss_history()
        # 允许数值误差
        self.assertAlmostEqual(res, 4.29, delta=0.5)

    def test_linear_model_with_scaler(self):
        x, y = linear_data(data_size=100000, seed=777)

        rescaled_x, scaler = standardization(x)

        model = LinearRegressionModel(niter=100, learning_rate=0.1, reg_param=0.1)
        model.fit(rescaled_x, y)

        # 测试点需要是2D数组
        test_point = np.array([[1, 1]])
        res = model.predict(scaler.rescale(test_point))[0]

        print("\nFinal Results:")
        print(f"Predicted: {res:.4f}")
        print(f"Weights: {model.weights}")
        print(f"Bias: {model.bias[0]:.4f}")

        model.plot_loss_history()
        # 允许数值误差
        self.assertAlmostEqual(res, 4.29, delta=0.5)  # predict with scaler: 4.2859, without scaler: 3.9075


if __name__ == "__main__":
    unittest.main()
