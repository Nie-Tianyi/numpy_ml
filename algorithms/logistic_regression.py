"""
logistic regression model
"""

import unittest

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from algorithms.activation_functions import sigmoid
from algorithms.gradient_descent import compute_gradient
from algorithms.loss_function import cross_entropy_loss
from algorithms.model_abstract import MachineLearningModel
from algorithms.regularization import Regularization, lasso, ridge, lasso_gradient, ridge_gradient
from algorithms.normaliser import z_score_normalisation
from test_data_set.test_data_gen import binary_data


class LogisticRegressionModel(MachineLearningModel):
	"""
	Logistic Regression Model
	"""

	def __init__(
		self,
		niter: int = 1000,
		learning_rate: float = 0.01,
		reg_param: float = 0.3,
		regularization=Regularization.RIDGE,
		threshold=0.5,
	):
		super().__init__(niter, learning_rate, reg_param, regularization)
		self.threshold = threshold

	def __init_weights_and_bias(self, dim: int):
		self.weights = np.random.randn(dim)
		self.bias = np.zeros(1)

	def predict_possibility(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
		"""
		:params x: 需要预测的数据， shape应该是(m,n)
		:returns: 返回模型预测值
		"""
		assert x.shape[1] == self.weights.shape[0]

		return sigmoid(np.dot(x, self.weights) + self.bias)

	def predict(self, x):
		"""
		预测数据，大于 self.threshold 的预测为1，其余为0
		:param x: 需要预测的数据，最好是二维NDArray
		:return: 返回一个一维的预测结果 0 或者是 1
		"""
		y_hat = self.predict_possibility(x)
		return (y_hat >= self.threshold).astype(float)

	def evaluate(self, x_test, y_test) -> float:
		"""
		评估模型性能，计算准确率
		:param x_test: 测试特征
		:param y_test: 测试标签（0/1）
		:return: 准确率 (0.0-1.0)
		"""
		# 确保y_test是整数类型（避免浮点比较问题）
		y_test = y_test.astype(int)

		# 预测并计算准确率
		y_hat = self.predict(x_test)
		res = (y_hat == y_test).astype(float)
		accuracy = np.mean(res)

		return float(accuracy)

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

		for _ in tqdm(range(self.niter)):
			y_hat = self.predict_possibility(x)

			# 计算记录损失
			loss = cross_entropy_loss(y_hat, y)
			(dlt_w, dlt_b) = compute_gradient(x, y_hat, y)
			match self.reg:
				case Regularization.LASSO:
					loss += lasso(self.weights, self.lambda_, m)
					dlt_w += lasso_gradient(self.weights, self.lambda_, m)
				case Regularization.RIDGE:
					loss += ridge(self.weights, self.lambda_, m)
					dlt_w += ridge_gradient(self.weights, self.lambda_, m)
				case Regularization.NO_REGULARIZATION:
					pass

			self.loss_history.append(loss)
			# 更新梯度
			self.weights -= self.lr * dlt_w
			self.bias -= self.lr * dlt_b


class Unittest(unittest.TestCase):
	def test_logistic_regression(self):
		(x, y) = binary_data(data_size=10000, seed=777)

		rescaled_x, scalar = z_score_normalisation(x)

		model = LogisticRegressionModel(niter=5000, learning_rate=0.1, reg_param=0.01)
		model.fit(rescaled_x, y)

		# 使用没有缩放过的数据训练的模型，作为对比
		model_no_scaled = LogisticRegressionModel(niter=5000, learning_rate=0.1, reg_param=0.01)
		model_no_scaled.fit(x, y)

		# 处于 x_1 + x_2 = 1 右边的点预测结果应该大于0.5，并且离决策边际越远，预测结果越接近于1
		test_point = np.array([[1, 1]])
		res = model.predict_possibility(scalar.rescale(test_point))[0]
		print("\nFinal Results:")
		print(f"Predicted: {res:.4f}")  # 0.9967，有99.67%的概率这个点是1
		print(f"Weights: {model.weights}")
		print(f"Bias: {model.bias[0]:.4f}")

		model.plot_loss_history(title="Loss History with Rescaling", label="Cross Entropy Loss")
		model_no_scaled.plot_loss_history(title="Loss History without Rescaling", label="Cross Entropy Loss")
		# 使用缩放后的数据训练的模型收敛的又快又好
		print(f"Rescaled model's final loss: {model.loss_history[-1]:.4f}")
		print(f"Un-rescaled model's final loss: {model_no_scaled.loss_history[-1]:.4f}")

		(test_x, test_y) = binary_data(data_size=10000, seed=1)
		rescaled_test_x = scalar.rescale(test_x)
		acc_rescaled = model.evaluate(rescaled_test_x, test_y)
		print("Rescaled model's Accuracy:", acc_rescaled)
		acc_unrescaled = model_no_scaled.evaluate(test_x, test_y)
		print("Un-rescaled model's Accuracy", acc_unrescaled)

		self.assertAlmostEqual(res, 1.0, delta=0.1)


if __name__ == "__main__":
	unittest.main()
