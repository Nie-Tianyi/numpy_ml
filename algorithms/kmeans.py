import unittest

from algorithms.distance_metrics import Metric, EuclidianDistance


class KMeans:
    def __init__(self, k=2, max_iter=100, distance_metrics: type[Metric] = EuclidianDistance):
        self.k = k
        self.max_iter = max_iter
        self.metrics = distance_metrics
        self.centroids = None

    def __init_centroids(self, x, seed = 0):
        """
        随机从x中挑选三个centroids
        :param x: 输入的数据
        :param seed: 随机数种子
        :return: None
        """
        pass

    def fit(self, x):
        pass


class Unittest(unittest.TestCase):
    def test_kmeans(self):
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
    