"""
Regularization terms
"""

from enum import Enum

import numba
import numpy as np


class Regularization(Enum):
    """
    Enum class for regularization terms
    """

    No_REGULARIZATION = 0
    LASSO = 1
    RIDGE = 2


@numba.njit(fastmath=True)
def lasso(weights, rg_param, m):
    return (rg_param / m) * np.sum(weights)


@numba.njit(fastmath=True)
def ridge(weights, rg_param, m):
    """
    L2 regularization term
    :param weights: model weights
    :param rg_param: hyperparameter
    :param m: number of data
    :return: loss brought by regularization
    """
    return (rg_param / 2 / m) * np.sum(weights**2)


@numba.njit
def ridge_gradient(weights, rg_param, m):
    """
    L2 regularization gradient term
    :param weights: model weights
    :param rg_param: hyperparameter
    :param m: number of data
    :return: gradient brought by regularization
    """
    return (rg_param / m) * weights
