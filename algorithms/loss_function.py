"""
Loss-functions pack
"""

import unittest
from enum import Enum

import numpy as np


class LossFunctions(Enum):
    """
    Enum class of loss functions
    """

    MEAN_SQUARED_ERROR = 1  # 线性回归
    CROSS_ENTROPY_LOSS = 2  # 二分类问题
    SPARSE_CROSS_ENTROPY_LOSS = 3  # 多分类问题


def mean_square_error(y_pred, y_true):
    r"""
    \frac{1}{m} * \sum (y_pred - y_true)^2
    y_predict & y should be the same shape
    :param y_pred: predicted value, a scalar or a numpy array
    :param y_true: real label value, a scalar or a numpy array
    :return: MSE loss
    """
    diff = y_pred - y_true
    # # 分解平方计算防止溢出
    # safe_diff = np.clip(diff, -1e4, 1e4)  # 限制差值范围
    return (1 / 2) * np.mean(diff**2)  # 使用分解后的安全值计算


def cross_entropy_loss(y_pred, y_true):
    """
    cross-entropy loss
    :param y_pred: predicted value, a scalar or a numpy array
    :param y_true: true label value, a scalar or a numpy array
    """
    # 添加数值稳定性处理，避免log(0)错误
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def sparse_cross_entropy_loss(y_pred, y_real):
    """
    sparse cross-entropy loss
    :param y_pred: predicted value, a scalar or a numpy array
    :param y_real: real label value, a scalar or a numpy array
    :return: loss value, a scalar or a numpy array
    """
    assert y_pred.shape == y_real.shape, "Unmatched shape between predicted value and real label"
    # y_pred.shape = y.shape = (m, k)
    epsilon = 1e-8  # 防止log(0)
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0 - epsilon)

    res = -y_real * np.log(y_pred_clipped)
    return np.sum(res)


class Unittest(unittest.TestCase):
    def test_mean_square_error(self):
        self.assertEqual(mean_square_error(np.array([1, 1, 1]), np.array([0, 0, 0])), 0.5)

    def test_cross_entropy_loss(self):
        loss1 = cross_entropy_loss(np.array([0.9988]), np.array([1.0]))  # 0.0012
        loss2 = cross_entropy_loss(np.array([0.0012]), np.array([1.0]))  # 6.7254
        self.assertLess(loss1, 0.1)
        self.assertGreater(loss2, 6)


if __name__ == "__main__":
    unittest.main()
