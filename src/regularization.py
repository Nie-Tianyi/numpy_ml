"""
Regularization terms
"""
from enum import Enum


class RegularizationTerm(Enum):
    """
    Enum class for regularization terms
    """
    L1_REGULARIZATION = 1
    L2_REGULARIZATION = 2