"""
Loss-functions pack
"""

import unittest

import numba
import numpy as np


@numba.njit
def mean_square_error(y_predict, y_real):
    r"""
    y_predict & y should be the same shape
    :param y_predict: predicted value, a scalar or a numpy array
    :param y_real: real label value, a scalar or a numpy array
    :return: MSE loss
    """
    return 0.5 * np.mean((y_predict ** 2 - y_real ** 2))


class Unittest(unittest.TestCase):
    def test_mean_square_error(self):
        assert mean_square_error(np.array([1, 1, 1]), np.array([0, 0, 0])), 1


if __name__ == "__main__":
    unittest.main()
