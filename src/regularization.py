"""
Regularization terms
"""
from enum import Enum

import numba
import numpy as np


class RegularizationTerm(Enum):
    """
    Enum class for regularization terms
    """
    No_REGULARIZATION = 0
    LASSO = 1
    RIDGE = 2


@numba.njit
def ridge(weights, rg_param, m):
    return (rg_param / 2 * m) * np.sum(weights ** 2)


@numba.njit
def ridge_gradient(weights, rg_param, m):
    return (rg_param / m) * weights
