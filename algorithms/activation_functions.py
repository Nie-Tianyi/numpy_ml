"""
常见激活函数
"""

import unittest

import numba
import numpy as np
from numpy.typing import NDArray

@numba.njit(parallel=True, fastmath=True)
def sigmoid(x):
	"""
	sigmoid function
	:param x: x, a scalar or a matrix
	:return: a scalar or a matrix
	"""
	return 1 / (1 + np.exp(-x))


# numba不支持axis属性，所以这个函数不能使用numba加速
def softmax(x: NDArray[np.float64], axis=0):
	x = x - np.max(x, axis=axis, keepdims=True)  # 平移防止溢出
	ex = np.exp(x)
	return ex / ex.sum(axis=axis, keepdims=True)


def relu(x):
	"""
	ReLU Rectified Linear Unit 激活函数
	:param x: x
	:return: ReLU(x)
	"""
	return np.max(0, x)


class Unittest(unittest.TestCase):
	def test_sigmoid(self):
		x = np.array([-1, -0.1, 0, 0.1, 1])
		print(sigmoid(x))
		self.assertTrue(
			np.allclose(
				sigmoid(x),
				np.array([0.26894142, 0.47502081, 0.5, 0.52497919, 0.73105858]),
			)
		)

	def test_softmax(self):
		x = np.array([[999, 1000, 1001]], dtype=np.float64)
		self.assertTrue(
			np.allclose(softmax(x, axis=1), np.array([[0.09003057, 0.24472847, 0.66524096]]))
		)


if __name__ == "__main__":
	unittest.main()
