import unittest
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from algorithms.distance_metrics import EuclidianDistance, Metric


class LossRecord:
    """
    记录每一次KMeans的损失
    """

    def __init__(self, centroids, loss):
        self.centroids = centroids
        self.loss = loss

    def __str__(self):
        return f"Centroids: {self.centroids}, Loss: {self.loss}"


class KMeans:
    """
    K-Means 算法
    """

    def __init__(
        self, k: int = 2, max_iter: int = 100, distance_metrics: type[Metric] = EuclidianDistance
    ):
        self.k = k  # 聚几类
        self.max_iter = max_iter  # 最多循环多少次
        self.metrics = distance_metrics  # 用什么衡量距离的算法，默认使用欧几里得距离
        self.centroids = None  # 中心点
        self.loss_history: List[LossRecord] = []  # 记录训练损失

    def __init_centroids(self, x: NDArray[np.float64], seed: int):
        """
        随机从x中挑选三个centroids
        :param seed: 随机数种子
        :return: None
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(x), size=self.k, replace=False)
        self.centroids = x[indices]

    def fit(self, x: NDArray[np.float64], random_state: Optional[int] = None):
        """
        开始聚类
        :param x: 需要聚类的数据
        :param random_state: 随机值，如果不设置则随机取值
        """
        (m, n) = x.shape
        if random_state is None:
            random_state = np.random.randint(1e6)

        self.__init_centroids(x, seed=random_state)  # 随机选择k个点作为centroid

        for _ in range(self.max_iter):
            # 计算每个点到centroid的距离，分组
            labels = self.predict(x)

            # 计算损失
            loss = 0
            for i in range(self.k):
                loss += self.metrics.distance(x[labels == i], self.centroids[i]).sum() / m
            self.loss_history.append(LossRecord(self.centroids.copy(), loss))

            # 更新centroid
            for i in range(self.k):
                self.centroids[i] = x[labels == i].mean(axis=0)

            # 检查是否收敛
            if len(self.loss_history) > 1 and np.allclose(
                self.centroids, self.loss_history[-1].centroids
            ):
                break

    def predict(self, x):
        """
        预测x所属的类别
        """
        (m, n) = x.shape()
        distances = np.zeros(shape=(m, self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:i] = self.metrics.distance(x, centroid)
        return np.argmin(distances, axis=1)


class Unittest(unittest.TestCase):
    def test_kmeans(self):
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
