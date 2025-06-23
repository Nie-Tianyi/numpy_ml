import unittest

import numpy as np


class Standardizer:
    def __init__(self, x: np.ndarray):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def rescale(self, x):
        return (x - self.mean) / self.std


class Unittest(unittest.TestCase):
    def test_standardizer(self):
        arr = np.arange(0, 10).reshape((5, 2))
        scaler = Standardizer(arr)
        rescaled_arr = scaler.rescale(arr)
        self.assertEqual(arr.shape, rescaled_arr.shape)


if __name__ == "__main__":
    unittest.main()
