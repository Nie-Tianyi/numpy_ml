"""
评估模型的方法，默认有 MSE，Accuracy，
"""
from abc import ABC, abstractmethod

import numpy as np

from algorithms.loss_function import mean_square_error


class EvaluationMethod(ABC):
    """
    评估模型的方法
    """
    @staticmethod
    @abstractmethod
    def evaluate(y_hat, y):
        """
        评估模型
        :param y_hat: 模型预测值
        :param y: 实际标签
        """
        pass


class MeanSquaredError(EvaluationMethod):
    @staticmethod
    def evaluate(y_hat, y):
        return mean_square_error(y_hat, y)


class Accuracy(EvaluationMethod):
    @staticmethod
    def evaluate(y_hat, y):
        return np.mean(y_hat == y)