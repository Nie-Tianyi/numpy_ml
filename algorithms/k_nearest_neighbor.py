"""
kNN 算法
"""

import numpy as np
from numpy.typing import NDArray

from algorithms.distance_metrics import Metric, EuclidianDistance


class KNearestNeighbor:
    def __init__(self, k: int, distance_metric: type[Metric] = EuclidianDistance):
        self.y = None
        self.x = None
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]):
        self.x = x
        self.y = y

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        预测x的标签
        """
        distances = self.distance_metric.distance(x, self.x)

        # 获取最近的k个邻居索引
        k_nearest_indices = np.argpartition(distances, self.k, axis=1)[:, : self.k]

        # 获取对应的标签
        k_nearest_labels = self.y[k_nearest_indices]

        # 统计每个样本的k个邻居中出现次数最多的标签
        labels = np.array(
            [np.bincount(sample_labels).argmax() for sample_labels in k_nearest_labels]
        )

        return labels
