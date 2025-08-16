"""
常见激活函数
"""

import unittest
from abc import ABC

import numba
import numpy as np


class ActivationFunction(ABC):
	@staticmethod
	def cal(x):
		pass

	@staticmethod
	def gradient(x):
		pass


class Sigmoid(ActivationFunction):
	"""Sigmoid 函数"""

	@staticmethod
	@numba.njit(parallel=True, fastmath=True)
	def cal(x):
		"""
		:param x: x, a scalar or a matrix
		:return: a scalar or a matrix
		"""
		return 1 / (1 + np.exp(-x))

	@staticmethod
	@numba.njit(parallel=True, fastmath=True)
	def gradient(x):
		"""
		:param x: shape (m, n) 的2D NDArray
		:return:
		"""
		return Sigmoid.cal(x)(1 - Sigmoid.cal(x))


class Softmax(ActivationFunction):
	"""Softmax 函数"""

	@staticmethod
	def cal(x):
		"""
		:param x: 应该是一个2D NDArray，默认沿着axis=1做Softmax计算
		:return:
		"""
		x = x - np.max(x, axis=1, keepdims=True)  # 平移防止溢出
		ex = np.exp(x)
		return ex / ex.sum(axis=1, keepdims=True)

	@staticmethod
	def gradient(x):
		pass


class ReLU(ActivationFunction):
	"""ReLU Rectified Linear Unit 激活函数"""

	@staticmethod
	def cal(x):
		"""
		:param x: x
		:return: ReLU(x)
		"""
		return np.maximum(x, 0)

	@staticmethod
	def gradient(x):
		"""
		ReLU's derivatives
		:param x: x, numpy array
		:return: x <=0 => 0; x > 0 => 1
		"""
		return np.greater(x, 0).astype(int)


class Unittest(unittest.TestCase):
	def test_sigmoid(self):
		x = np.array([-1, -0.1, 0, 0.1, 1])
		self.assertTrue(
			np.allclose(
				Sigmoid.cal(x),
				np.array([0.26894142, 0.47502081, 0.5, 0.52497919, 0.73105858]),
			)
		)

	def test_softmax(self):
		x = np.array([[999, 1000, 1001]], dtype=np.float64)
		self.assertTrue(
			np.allclose(Softmax.cal(x), np.array([[0.09003057, 0.24472847, 0.66524096]]))
		)

	def test_relu(self):
		x = np.array([[-0.1, 1, 2], [1, -1, 0.4]], dtype=np.float64)
		self.assertTrue(
			np.allclose(ReLU.cal(x), np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 0.4]]))
		)
		self.assertTrue(np.allclose(ReLU.gradient(x), np.array([[0, 1, 1], [1, 0, 1]])))


if __name__ == "__main__":
	unittest.main()
