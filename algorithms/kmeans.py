import unittest
from typing import List, Optional

import numpy as np

from algorithms.distance_metrics import EuclidianDistance, Metric


class LossRecord:
    """
    记录每一次KMeans的损失
    """
    def __init__(self, centroid, loss):
        self.centroid = centroid
        self.loss = loss


class KMeans:
    """
    K-Means 算法
    """
    def __init__(self, x, k=2, max_iter=100, distance_metrics: type[Metric] = EuclidianDistance):
        self.x = x # 需要聚类的数据
        self.k = k # 聚几类
        self.max_iter = max_iter # 最多循环多少次
        self.metrics = distance_metrics # 用什么衡量距离的算法，默认使用欧几里得距离
        self.centroids = None # 中心点
        self.loss_history: List[LossRecord] = [] # 记录训练损失

    def __init_centroids(self, seed: Optional[int] = None):
        """
        随机从x中挑选三个centroids
        :param seed: 随机数种子
        :return: None
        """
        if seed is None:
            seed = np.random.randint(1e6)

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.x), size=self.k, replace=False)
        self.centroids = self.x[indices]




    def fit(self):
        """
        开始聚类
        """
        self.__init_centroids()

        for _ in range(self.max_iter):
            pass




class Unittest(unittest.TestCase):
    def test_kmeans(self):
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
    