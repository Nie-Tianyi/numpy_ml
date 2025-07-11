"""
重新缩放训练数据
"""

import unittest

import numpy as np


class Standardizer:
    """
    Z-score Normalisation scaler
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def rescale(self, x):
        """
        使用Z-score Normalisation 重新缩放数据
        :param x: 需要缩放的数据
        :return: 缩放后的数据
        """
        return (x - self.mean) / self.std


def standardization(x):
    """
    使用z-score Normalisation 重新缩放数据，返回缩放后的数据
    :param x: 需要缩放的数据
    :return: 返回缩放后的数据，以及一个缩放器，用于缩放验证数据
    """
    scaler = Standardizer(np.mean(x, axis=0), np.std(x, axis=0))
    return scaler.rescale(x), scaler


class Unittest(unittest.TestCase):
    def test_standardizer(self):
        arr = np.arange(0, 10).reshape((5, 2))
        rescaled_arr, scaler = standardization(arr)
        self.assertEqual(arr.shape, rescaled_arr.shape)


if __name__ == "__main__":
    unittest.main()
