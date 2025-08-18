"""
Regularization terms
"""

from abc import ABC, abstractmethod

import numpy as np


class Regularization(ABC):
    """正则化基类"""

    @staticmethod
    @abstractmethod
    def loss(weights, rg_param, m):
        """计算正则化项带来的损失，返回一个标量"""
        pass

    @staticmethod
    @abstractmethod
    def derivative(weights, rg_param, m):
        """计算正则化项带来的梯度，一般是一个跟 weights 形状相同的矩阵"""
        pass


class NoReg(Regularization):
    """表示不适用任何正则化手段，用来占位"""

    @staticmethod
    def loss(weights, rg_param, m):
        return 0

    @staticmethod
    def derivative(weights, rg_param, m):
        return np.zeros_like(weights)


class LASSO(Regularization):
    """L1 正则化项"""

    @staticmethod
    def loss(weights, rg_param, m):
        return (rg_param / m) * np.sum(weights)

    @staticmethod
    def derivative(weights, rg_param, m):
        return (rg_param / m) * np.sign(weights)


class Ridge(Regularization):
    """L2 正则化"""

    @staticmethod
    def loss(weights, rg_param, m):
        return (rg_param / 2 / m) * np.sum(weights**2)

    @staticmethod
    def derivative(weights, rg_param, m):
        return (rg_param / m) * weights
