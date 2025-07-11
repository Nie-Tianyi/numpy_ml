"""
Loss-functions pack
"""
import unittest
from enum import Enum

import numba
import numpy as np


class LossFunctions(Enum):
    """
    Enum class of loss functions
    """
    MEAN_SQUARED_ERROR = 1  # 线性回归
    CROSS_ENTROPY_LOSS = 2  # 二分类问题
    SPARSE_CROSS_ENTROPY_LOSS = 3  # 多分类问题


@numba.njit(fastmath=True)
def mean_square_error(y_pred, y_true):
    r"""
    \frac{1}{m} * \sum (y_pred - y_true)^2
    y_predict & y should be the same shape
    :param y_pred: predicted value, a scalar or a numpy array
    :param y_true: real label value, a scalar or a numpy array
    :return: MSE loss
    """
    return 0.5 * np.mean((y_pred - y_true) ** 2)


@numba.njit(fastmath=True)
def cross_entropy_loss(y_pred, y_true):
    """
    cross-entropy loss
    :param y_pred: predicted value, a scalar or a numpy array
    :param y_true: true label value, a scalar or a numpy array
    """
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


class Unittest(unittest.TestCase):
    def test_mean_square_error(self):
        self.assertEqual(mean_square_error(np.array([1, 1, 1]), np.array([0, 0, 0])), 0.5)


if __name__ == "__main__":
    unittest.main()
