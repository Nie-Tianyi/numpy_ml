import unittest

import numpy as np
from matplotlib import pyplot as plt


def cluster_points_data(k=3, data_size=1000, seed=1, min_distance=5):
    """
    生成k个簇的点数据
    :param min_distance: 中心点的最小距离
    :param k: 簇的数量
    :param data_size: 每个簇的点的数量
    :param seed: 随机种子
    :return:
    """
    np.random.seed(seed)

    # 初始化所有簇的点
    data = np.empty((0, 2))
    # 初始化所有簇的标签
    labels = np.empty((0, 1))

    for i in range(k):
        # 随机生成簇的中心
        center = np.random.randn(2) * min_distance
        # 随机生成簇的点
        points = np.random.randn(data_size, 2) + center
        # 生成标签
        label = np.full(data_size, i)
        # 合并所有簇的点
        data = np.vstack((data, points))
        # 合并所有簇的标签
        labels = np.vstack((labels, label.reshape(-1, 1)))

    # 打乱数据
    shuffle_index = np.random.permutation(data.shape[0])
    data = data[shuffle_index]
    labels = labels[shuffle_index]

    return data, labels


class UnitTest(unittest.TestCase):
    def test_cluster_points_data(self):
        data, labels = cluster_points_data(k=4, data_size=150, seed=135, min_distance=3)

        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.show()

        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
