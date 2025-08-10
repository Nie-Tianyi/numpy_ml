"""
Regularization terms
"""

from enum import Enum

import numpy as np


class Regularization(Enum):
	"""
	Enum class for regularization terms
	"""

	NO_REGULARIZATION = 0
	LASSO = 1
	RIDGE = 2


def lasso(weights, rg_param, m):
	return (rg_param / m) * np.sum(weights)


def ridge(weights, rg_param, m):
	"""
	L2 regularization term
	:param weights: model weights
	:param rg_param: hyperparameter
	:param m: amount of data
	:return: loss brought by regularization
	"""
	return (rg_param / 2 / m) * np.sum(weights**2)


def ridge_gradient(weights, rg_param, m):
	"""
	L2 regularization gradient term
	:param weights: model weights
	:param rg_param: hyperparameter
	:param m: number of data
	:return: gradients brought by regularization
	"""
	return (rg_param / m) * weights
