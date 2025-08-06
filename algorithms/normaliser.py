"""
重新缩放训练数据
"""

import unittest
from enum import Enum

import numpy as np


class Normaliser(Enum):
    NoRescale = 0
    ZScore = 1
    MaxMin = 2


class ZScoreNormaliser:
    """
    Z-score Normalisation scaler
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.std[std == 0] = 1

    def rescale(self, x):
        """
        使用Z-score Normalisation 重新缩放数据
        :param x: 需要缩放的数据
        :return: 缩放后的数据
        """
        return (x - self.mean) / self.std


def z_score_normalisation(x):
    """
    使用z-score Normalisation 重新缩放数据，返回缩放后的数据
    如果数据的标准差为0，则不做任何缩放
    :param x: 需要缩放的数据
    :return: 返回缩放后的数据，以及一个缩放器，用于缩放验证数据
    """
    scaler = ZScoreNormaliser(np.mean(x, axis=0), np.std(x, axis=0))
    return scaler.rescale(x), scaler


class Unittest(unittest.TestCase):
    def test_normaliser(self):
        arr = np.arange(0, 10).reshape((5, 2))
        print("before normalise:", arr)
        rescaled_arr, scaler = z_score_normalisation(arr)
        print("after normalise:", rescaled_arr)
        self.assertEqual(arr.shape, rescaled_arr.shape)

    def test_normaliser_with_zeros(self):
        arr = np.zeros((5, 2), dtype=np.float64)
        print("before normalise:", arr)
        rescaled_arr, scaler = z_score_normalisation(arr)
        print("mean, std", scaler.mean, scaler.std)
        print("after normalise:", rescaled_arr)
        self.assertEqual(arr.shape, rescaled_arr.shape)
        np.testing.assert_almost_equal(rescaled_arr, arr)


if __name__ == "__main__":
    unittest.main()
