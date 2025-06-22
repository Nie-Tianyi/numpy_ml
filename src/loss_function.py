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
    MEAN_SQUARED_ERROR = 1
    CROSS_ENTROPY_LOSS = 2
    SPARSE_CROSS_ENTROPY_LOSS = 3


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


class Unittest(unittest.TestCase):
    def test_mean_square_error(self):
        self.assertEqual(mean_square_error(np.array([1, 1, 1]), np.array([0, 0, 0])), 0.5)


if __name__ == "__main__":
    unittest.main()
