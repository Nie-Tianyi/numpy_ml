import numba
import numpy as np


@numba.njit(fastmath=True)
def sigmoid(x):
    """
    sigmoid function
    :param x: x, a scalar or a matrix
    :return: a scalar or a matrix
    """
    return 1 / (1 + np.exp(-x))