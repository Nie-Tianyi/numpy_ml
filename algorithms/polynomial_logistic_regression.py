import unittest
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt

from numpy.typing import NDArray
from tqdm import tqdm

from algorithms.activation_functions import Softmax
from algorithms.gradient_descent import compute_gradient
from algorithms.loss_function import sparse_cross_entropy_loss
from algorithms.model_abstract import MachineLearningModel
from algorithms.normaliser import z_score_normalisation
from test_data_set.mnist import mnist
from test_data_set.test_data_gen import binary_data
from algorithms.regularization import Regularization, lasso, ridge, lasso_gradient, ridge_gradient


class PolynomialLogisticRegression(MachineLearningModel):
	"""
	多项式逻辑回归，处理多分类问题
	"""

	labels: Optional[NDArray[np.float64]]

	def __init__(
		self,
		niter=1000,
		learning_rate: float = 1,
		reg_param=0.03,
		regularization=Regularization.RIDGE,
	):
		super().__init__(niter, learning_rate, reg_param, regularization)
		self.labels = None

	def __init_weights_and_bias(self, dim: int, k: int):
		# dim 数据有多少个维度；k 多分类问题里面有多少个预测分类
		self.weights = np.random.rand(dim, k)
		self.bias = np.zeros(k)

	def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]):
		self.labels = np.unique(y)
		print("possible labels:", self.labels)
		# 把 y 从标签转换成 one-hot 编码 e.g. 9 => [0,0,0,0,0,0,0,0,0,1]
		y = (y.reshape(-1, 1) == self.labels).astype(np.float64)

		k = len(self.labels)
		(m, n) = x.shape

		self.__init_weights_and_bias(n, k)
		for _ in tqdm(range(self.niter)):
			y_pred = self.predict_possibility(x)

			# 计算记录损失
			loss = sparse_cross_entropy_loss(y_pred, y)
			(dlt_w, dlt_b) = compute_gradient(x, y_pred, y)
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

	def predict_possibility(self, x):
		"""
		预测概率，返回一个softmax处理后的概率NDArray
		:param x: 训练数据
		:return: 一个softmax处理后的概率NDArray，例如 [[0.1, 0.2, 0.7]]
		"""
		if self.weights is None or self.bias is None:
			raise ValueError("Model has not been initialised yet")
		# x.shape = (m, n) self.weights.shape = (n, K)
		z = np.dot(x, self.weights) + self.bias
		return Softmax.cal(z)

	def predict(self, x):
		poss = self.predict_possibility(x)
		return self.labels[np.argmax(poss, axis=1)]

	def evaluate(self, x_test, y_test) -> float:
		"""
		评估模型性能，计算准确率
		:param x_test: 测试特征
		:param y_test: 测试标签（0/1）
		:return: 准确率 (0.0-1.0)
		"""
		# 预测并计算准确率
		y_hat = self.predict(x_test)
		accuracy = np.mean(y_hat == y_test)
		return float(accuracy)


class Unittest(unittest.TestCase):
	def test_polynomial_logistic_regression(self):
		(x, y) = binary_data(data_size=10000, seed=7)  # 多分类问题当然也能处理二分类问题啦
		# 训练模型
		model = PolynomialLogisticRegression(niter=1000, learning_rate=0.1, reg_param=0.01)
		x, scaler = z_score_normalisation(x)
		model.fit(x, y)
		# 测试数据
		test_point = np.array([[1, 1]])
		res = model.predict_possibility(test_point)
		# 检验结果
		model.plot_loss_history(label="Sparse Cross Entropy Loss")
		self.assertGreaterEqual(res[0, 1], 0.9)  # 确保类别1的概率 > 90%

		test_x, test_y = binary_data(data_size=1000, seed=138)
		test_x = scaler.rescale(test_x)
		acc = model.evaluate(test_x, test_y)
		print("Accuracy:", acc)

	def test_mnist(self):
		(x, y) = mnist(data_size=70000, seed=7)

		def reshape_x(arr):
			# 将 n*28*28 的 image 拍平成 n*784 的二维数组
			return arr.reshape(arr.shape[0], -1)

		x = reshape_x(x)
		x, scaler = z_score_normalisation(x)
		model = PolynomialLogisticRegression(niter=100, learning_rate=1, reg_param=0.01)
		model.fit(x, y)

		model.plot_loss_history(label="Sparse Cross Entropy Loss")

		# 测试数据
		(test_x, test_y) = mnist(data_size=1, seed=138)
		reshaped_test_x = reshape_x(test_x)
		rescaled_test_x = scaler.rescale(reshaped_test_x)
		y_hat = model.predict(rescaled_test_x)

		plt.imshow(test_x[0], cmap="grey")
		plt.title(f"Predicted Label: {y_hat[0]}, Real Label: {test_y[0]}")
		plt.show()

		self.assertEqual(test_y[0], y_hat[0])

	def test_mnist_accuracy(self):
		(x, y) = mnist(data_size=70000, seed=7)

		def reshape_x(arr):
			return arr.reshape(arr.shape[0], -1)

		x = reshape_x(x)
		x, scaler = z_score_normalisation(x)

		# 将数据分成 60000 的训练集和 10000 的测试集
		x_train, x_test = x[:60000], x[60000:]
		y_train, y_test = y[:60000], y[60000:]

		model = PolynomialLogisticRegression(niter=10000, learning_rate=0.2, reg_param=0.01)
		model.fit(x_train, y_train)
		model.plot_loss_history()

		acc = model.evaluate(x_test, y_test)
		print("Accuracy:", acc)  # 0.9153
		self.assertGreaterEqual(acc, 0.9)  # 确保准确率 > 90%


if __name__ == "__main__":
	unittest.main()
