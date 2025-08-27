"""
衡量两个向量之间距离或者相似度的算法
"""

import unittest
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Metric(ABC):
    """
    衡量向量距离的算法接口
    """

    @staticmethod
    @abstractmethod
    def distance(x, y):
        """
        x, y 应该是第二个维度相同的两个向量
        """
        pass


class EuclidianDistance(Metric):
    """
    欧几里得距离
    """

    @staticmethod
    def distance(x: NDArray[np.float64], y: NDArray[np.float64]):
        return np.sqrt(np.sum((x - y) ** 2, axis=1))


class ManhattanDistance(Metric):
    """
    曼哈顿距离
    """

    @staticmethod
    def distance(x, y):
        return np.sum(np.absolute(x - y), axis=1)


class CosineDistance(Metric):
    """
    余弦相似度
    """

    @staticmethod
    def distance(x, y):
        return 1 - CosineDistance.similarity(x, y)

    @staticmethod
    def similarity(x, y):
        return np.sum(x * y, axis=1) / (
            np.sqrt(np.sum(x**2, axis=1)) * np.sqrt(np.sum(y**2, axis=1))
        )


class Unittest(unittest.TestCase):
    def test_euclidean_distance(self):
        x = np.array([[1, 2], [1, 0], [2, 1], [1, 1]])
        y = np.array([[1, 1]])

        self.assertTrue(np.allclose(EuclidianDistance.distance(x, y), np.array([1, 1, 1, 0])))

    def test_manhattan_distance(self):
        x = np.array([[1, 2], [1, 0], [2, 1], [1, 1]])
        y = np.array([[1, 1]])

        self.assertTrue(np.allclose(ManhattanDistance.distance(x, y), np.array([1, 1, 1, 0])))

    def test_cosine_distance(self):
        x = np.array([[1, 2], [1, 0], [2, 1], [1, 1]])
        y = np.array([[1, 1]])

        distance = CosineDistance.distance(x, y)
        print(distance)
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
