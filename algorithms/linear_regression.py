"""
Fixed Linear Regression Model
"""

import unittest

import numba
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from algorithms.loss_function import mean_square_error
from algorithms.model_abstract import MachineLearningModel
from algorithms.regularization import Regularization, lasso, ridge
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
		regularization=Regularization.RIDGE,
	):
		super().__init__(niter, learning_rate, reg_param, regularization)

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

			# 更新梯度 (包含正则化)
			if self.reg == Regularization.NO_REGULARIZATION:
				loss = mean_square_error(y_hat, y)
				self.loss_history.append(loss)

				(dlt_w, dlt_b) = self.__compute_gradient_without_regularization(x, y_hat, y, m)
				self.weights -= self.lr * dlt_w
				self.bias -= self.lr * float(dlt_b)
			elif self.reg == Regularization.LASSO:
				loss = mean_square_error(y_hat, y) + lasso(self.weights, self.lambda_, m)
				self.loss_history.append(loss)

				(dlt_w, dlt_b) = self.__computer_gradient_with_l1_regularization(
					x, y_hat, y, m, self.lambda_, self.weights
				)
				self.weights -= self.lr * dlt_w
				self.bias -= self.lr * float(dlt_b)
			elif self.reg == Regularization.RIDGE:
				loss = mean_square_error(y_hat, y) + ridge(self.weights, self.lambda_, m)
				self.loss_history.append(loss)

				(dlt_w, dlt_b) = self.__computer_gradient_with_l2_regularization(
					x, y_hat, y, m, self.lambda_, self.weights
				)
				self.weights -= self.lr * dlt_w
				self.bias -= self.lr * float(dlt_b)

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

	@staticmethod
	@numba.jit(fastmath=True)
	def __compute_gradient_without_regularization(
		x: NDArray[np.float64],
		y_pred: NDArray[np.float64],
		y_real: NDArray[np.float64],
		m: int,
	) -> tuple[NDArray[np.float64], np.floating]:
		error = y_pred - y_real
		dlt_w = np.dot(x.T, error) / m
		dlt_b = np.mean(error)

		return dlt_w, dlt_b

	@staticmethod
	@numba.njit(fastmath=True)
	def __computer_gradient_with_l2_regularization(
		x: NDArray[np.float64],
		y_pred: NDArray[np.float64],
		y_real: NDArray[np.float64],
		m: int,
		lambda_: float,
		weights: NDArray[np.float64],
	) -> tuple[NDArray[np.float64], np.floating]:
		error = y_pred - y_real
		dlt_w = np.dot(x.T, error) / m
		dlt_b = np.mean(error)
		dlt_w += (lambda_ / m) * weights

		return dlt_w, dlt_b

	@staticmethod
	@numba.njit(fastmath=True)
	def __computer_gradient_with_l1_regularization(
		x: NDArray[np.float64],
		y_pred: NDArray[np.float64],
		y_real: NDArray[np.float64],
		m: int,
		lambda_: float,
		weights: NDArray[np.float64],
	) -> tuple[NDArray[np.float64], np.floating]:
		error = y_pred - y_real
		dlt_w = np.dot(x.T, error) / m
		dlt_b = np.mean(error)
		dlt_w += (lambda_ / m) * np.sign(weights)

		return dlt_w, dlt_b


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

		model.plot_loss_history()

		# 允许数值误差
		self.assertAlmostEqual(
			res, 4.29, delta=0.5
		)  # predict with scaler: 4.2859, without a scaler: 3.9075


if __name__ == "__main__":
	unittest.main()
