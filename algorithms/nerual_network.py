import unittest

import numpy as np

from algorithms.activation_functions import relu
from algorithms.model_abstract import MachineLearningModel
from algorithms.regularization import Regularization


class NeuralNetwork(MachineLearningModel):
	def __init__(
		self,
		layers,
		niter=1000,
		learning_rate=0.1,
		reg_param: float = 0.3,
		regularization=Regularization.RIDGE,
	):
		super().__init__(niter, learning_rate, reg_param, regularization)
		self.layers = layers

	def fit(self, x, y):
		pass

	def predict(self, x):
		pass

	def evaluate(self, x_test, y_test) -> float:
		pass


class LinearLayer:
	def __init__(self, num, activation_function=relu):
		self.num = num  # 一个线性层
		self.activation_function = activation_function  # 激活函数，默认是ReLU
		self.weights = None
		self.bias = None

	def _init_weights_and_bias(self, dim):
		self.weights = np.random.randn(self.num, dim)
		self.bias = np.zeros(self.num)

	def forward(self, x):
		"""
		前向传播
		:param x: 输入，形状为 (m, dim)
		:return: 输出，形状为 (m, num)
		"""
		# weights.shape = (num, dim) x.shape = (n, dim)
		z = np.dot(x, self.weights.T) + self.bias
		# z.shape = (n, num)
		a = self.activation_function(z)
		return a


class Unittest(unittest.TestCase):
	def test_neural_network(self):
		self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
	unittest.main()
