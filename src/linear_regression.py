"""
Fixed Linear Regression Model
"""
import unittest

import numpy as np
import seaborn
from matplotlib import pyplot as plt

from src.loss_function import mean_square_error
from src.regularization import RegularizationTerm


class LinearRegressionModel:
    """
    Linear Regression Model (Fixed)
    """

    def __init__(self, niter=1000, learning_rate=0.01, regula_param=0.3):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.lambda_ = regula_param
        self.niter = niter
        self.loss_history = []

    def fit(self, x, y, regularization=RegularizationTerm.RIDGE):
        """
        fit function
        :param x: training data
        :param y: labels
        :param regularization: regularization term
        """
        assert x.shape[0] == y.shape[0], "x and y must be the same length"
        (m, dim) = x.shape
        y = y.flatten()

        self._init_weights_and_bias(dim)

        for i in range(self.niter):
            y_hat = self.predict(x)
            loss = mean_square_error(y_hat, y)
            self.loss_history.append(loss)

            # 更新梯度 (包含正则化)
            self._compute_gradient(x, y_hat, y, m, regularization)

            # 每100次迭代打印进度
            if i % 10 == 0:
                print(f"Iteration {i}: Loss={loss:.4f}, Weights={self.weights}, Bias={self.bias[0]:.4f}")

    def predict(self, x):
        """
        Predict values for x
        :param x:  should be the same length as weights
        :return: float predicted value
        """
        return np.dot(x, self.weights) + self.bias

    def _init_weights_and_bias(self, dim):
        # 初始化权重和偏置
        self.weights = np.random.randn(dim) * 0.01
        self.bias = np.zeros(1)

    def _compute_gradient(self, x, y_pred, y_real, m, regularization):
        # 计算基础梯度
        error = y_pred - y_real
        dlt_w = np.dot(x.T, error) / m
        dlt_b = np.mean(error)

        # 添加正则化梯度 (关键修复)
        if regularization == RegularizationTerm.RIDGE:
            dlt_w += (self.lambda_ / m) * self.weights
        elif regularization == RegularizationTerm.LASSO:
            dlt_w += (self.lambda_ / m) * np.sign(self.weights)

        # 更新参数
        self.weights -= self.lr * dlt_w
        self.bias -= self.lr * dlt_b


class Unittest(unittest.TestCase):
    def test_linear_model(self):
        data_size = 1000
        np.random.seed(777)
        x_1 = np.random.rand(data_size)
        x_2 = np.random.rand(data_size)
        noise = np.random.randn(data_size)  # 创建1维噪声数组

        # 创建特征矩阵 (1000, 2)
        x = np.vstack([x_1, x_2]).T

        # 创建目标值 (1000,)
        y = 0.99 * x_1 + 2.3 * x_2 + noise + 1

        model = LinearRegressionModel(niter=100, learning_rate=0.1, regula_param=0.1)
        model.fit(x, y)

        # 测试点需要是2D数组
        test_point = np.array([[1, 1]])
        res = model.predict(test_point)[0]

        print("\nFinal Results:")
        print(f"Predicted: {res:.4f}")
        print(f"Weights: {model.weights}")
        print(f"Bias: {model.bias[0]:.4f}")

        # 绘制损失曲线
        seaborn.lineplot(model.loss_history)
        plt.title("Training Loss History")
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.show()

        # 允许数值误差
        self.assertAlmostEqual(res, 4.29, delta=0.5)


if __name__ == "__main__":
    unittest.main()
