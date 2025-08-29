"""
K-Means clustering algorithm
"""

import unittest
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from algorithms.distance_metrics import EuclidianDistance, Metric
from test_data_set.cluster_points import cluster_points_data


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

    def __init_centroids_with_kmeans_pp(self, x: NDArray[np.float64], seed: int):
        """
        用KMeans++初始化centroids
        :param x: 输入数据
        :param seed: 随机数种子
        :return: None
        """
        (m, n) = x.shape
        indices = []
        rng = np.random.default_rng(seed)

        for i in range(self.k):
            if i == 0:
                indices.append(rng.choice(len(x), size=1, replace=False))
            else:
                distances = np.zeros(shape=(m, i))
                for j in range(i):
                    distances[:, j] = self.metrics.distance(x, x[indices[j]])
                min_distance = distances.min(axis=1)
                min_distance **= 2
                p = min_distance / min_distance.sum()
                indices.append(rng.choice(len(x), size=1, replace=False, p=p))

        self.centroids = x[indices]

    def fit(
        self, x: NDArray[np.float64], random_state: Optional[int] = None, use_kmeans_pp: bool = True
    ):
        """
        开始聚类
        :param x: 需要聚类的数据
        :param random_state: 随机值，如果不设置则随机取值
        :param use_kmeans_pp: 是否用KMeans++初始化centroids，默认使用
        :return: None
        """
        (m, n) = x.shape
        if random_state is None:
            random_state = np.random.randint(1e6)

        if use_kmeans_pp:
            self.__init_centroids_with_kmeans_pp(x, seed=random_state)
        else:
            self.__init_centroids(x, seed=random_state)  # 随机选择k个点作为centroid

        for _ in range(self.max_iter):
            # 计算每个点到centroid的距离，分组
            labels = self.predict(x)

            # 计算损失
            loss = 0
            for i in range(self.k):
                loss += self.metrics.distance(x[labels == i], self.centroids[i]).sum()
            loss /= m
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
        (m, n) = x.shape
        distances = np.zeros(shape=(m, self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = self.metrics.distance(x, centroid)
        return np.argmin(distances, axis=1)

    @property
    def final_loss(self):
        return self.loss_history[-1].loss


def exploring_data_with_kmeans(
    data,
    k_range,
    random_state_range: Optional[List[int]] = None,
    distance_metric: type[Metric] = EuclidianDistance,
    max_iter=1000,
):
    """
    用 KMeans 算法探索数据，尝试不同的K值，画出损失随K值变化曲线
    :param data: 需要被探索的数据
    :param k_range: 尝试的k值
    :param random_state_range: 每个K值会尝试多个random state以尝试找到全局最小值
    :param distance_metric: 用什么算法衡量向量距离，默认使用 EuclideanDistance
    :param max_iter: 每个KMeans算法最多循环多少次
    :return:
    """
    if random_state_range is None:
        random_state_range = range(10)

    ultimate_final_losses = []
    for k in tqdm(k_range):
        final_losses = []
        for random_state in random_state_range:
            kmeans = KMeans(k, max_iter, distance_metric)
            kmeans.fit(data, random_state)
            final_losses.append(kmeans.final_loss)
        ultimate_final_losses.append(min(final_losses))

    plt.plot(k_range, ultimate_final_losses)
    plt.title(label="Losses variation with different K")
    plt.xlabel("K value")
    plt.ylabel("Distortion Loss")
    plt.show()


class Unittest(unittest.TestCase):
    def test_kmeans(self):
        data, real_label = cluster_points_data(k=3, data_size=150, seed=135, min_distance=10)
        model = KMeans(k=3, max_iter=1000, distance_metrics=EuclidianDistance)
        model.fit(data, random_state=9)
        predicted_label = model.predict(data)

        plt.scatter(data[:, 0], data[:, 1], c=real_label)
        plt.title(label="Real Cluster")
        plt.show()
        plt.scatter(data[:, 0], data[:, 1], c=predicted_label)
        plt.title(label="Predicted Cluster")
        plt.show()  # 很容易卡在Local Minimum啊

        self.assertEqual(1 + 1, 2)

    def test_explore(self):
        data, real_label = cluster_points_data(k=6, data_size=150, seed=135, min_distance=10)
        exploring_data_with_kmeans(data, k_range=range(1, 10))

        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
