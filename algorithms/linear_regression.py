"""
Fixed Linear Regression Model
"""

import unittest

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from algorithms.gradient_descent import compute_gradient
from algorithms.loss_function import mean_square_error
from algorithms.model_abstract import MachineLearningModel
from algorithms.regularization import Regularization, Ridge
from algorithms.normaliser import z_score_normalisation
from test_data_set.test_data_gen import linear_data


class LinearRegressionModel(MachineLearningModel):
	"""
	Linear Regression Model
	"""

	def __init__(
		self,
		niter=1000,
		learning_rate=0.01,
		reg_param=0.3,
		regularization: Regularization = Ridge,
	):
		super().__init__(regularization, niter, learning_rate, reg_param)

	def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]):
		"""
		训练模型，x对应着数据，y对应着label，regularization代表正则化方式
		:param x: 假设是一个 (m,n) shape的 numpy.ndarray，m表示有多少数据，n表示数据的维度
		:param y: labels，应该是一个 (m,1) shape的 numpy.ndarray
		"""
		assert x.shape[0] == y.shape[0], "x and y must be the same length"
		(m, dim) = x.shape
		y = y.flatten()

		self.__init_weights_and_bias(dim)
		assert self.weights is not None and self.bias is not None, (
			"Weights and bias must be initialized"
		)

		for _ in tqdm(range(self.niter)):
			y_hat = self.predict(x)
			# 计算记录损失
			loss = mean_square_error(y_hat, y)
			(dlt_w, dlt_b) = compute_gradient(x, y_hat, y)

			loss += self.reg.loss(self.weights, self.lambda_, m)
			dlt_w += self.reg.derivative(self.weights, self.lambda_, m)

			self.loss_history.append(loss)
			# 更新梯度
			self.weights -= self.lr * dlt_w
			self.bias -= self.lr * dlt_b

	def predict(self, x: NDArray[np.float64]):
		"""
		预测数据
		:param x:  should be the same length as weights
		:return: float predicted value
		"""
		if self.weights is None:
			raise ValueError("Model has not been initialised yet")
		assert x.shape[1] == self.weights.shape[0]
		return np.dot(x, self.weights) + self.bias

	def evaluate(self, x_test, y_test) -> float:
		"""
		评估模型，返回测试数据集上的 Mean Square Error
		:param x_test: 测试数据x
		:param y_test: 测试数据y
		:return: 返回 MSE
		"""
		y_hat = self.predict(x_test)
		return mean_square_error(y_hat, y_test)

	def __init_weights_and_bias(self, dim: int):
		# 初始化权重和偏置
		self.weights = np.random.rand(dim)
		self.bias = np.zeros(1)


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
		model.plot_loss_history(label="Mean Square Error")
		# 允许数值误差
		self.assertAlmostEqual(res, 4.29, delta=0.5)

	def test_linear_model_with_scaler(self):
		x, y = linear_data(data_size=100000, seed=777)

		rescaled_x, scaler = z_score_normalisation(x)

		model = LinearRegressionModel(niter=100, learning_rate=0.1, reg_param=0.1)
		model.fit(rescaled_x, y)

		# 测试点需要是2D数组
		test_point = np.array([[1, 1]])
		res = model.predict(scaler.rescale(test_point))[0]

		print("\nFinal Results:")
		print(f"Predicted: {res:.4f}")
		print(f"Weights: {model.weights}")
		print(f"Bias: {model.bias[0]:.4f}")

		model.plot_loss_history(title="Loss History", label="Mean Square Error")

		# 允许数值误差
		self.assertAlmostEqual(
			res, 4.29, delta=0.5
		)  # predict with scaler: 4.2859, without a scaler: 3.9075


if __name__ == "__main__":
	unittest.main()
