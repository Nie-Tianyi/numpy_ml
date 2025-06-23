import unittest

import numpy as np


class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def rescale(self, x):
        return (x - self.mean) / self.std


def standardization(x):
    scaler = Standardizer(np.mean(x, axis=0), np.std(x, axis=0))
    return scaler.rescale(x), scaler


class Unittest(unittest.TestCase):
    def test_standardizer(self):
        arr = np.arange(0, 10).reshape((5, 2))
        rescaled_arr, scaler = standardization(arr)
        self.assertEqual(arr.shape, rescaled_arr.shape)


if __name__ == "__main__":
    unittest.main()
