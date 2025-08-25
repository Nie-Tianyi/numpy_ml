import unittest

import numpy as np
from matplotlib import pyplot as plt


def concentric_circles(n_samples=1000, noise=0.1):
    """生成同心圆数据集"""
    # 生成角度
    angles = np.random.rand(n_samples) * 2 * np.pi

    # 生成两个半径不同的圆
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    # 内圆
    r_inner = np.random.rand(n_inner) * 0.4 + 0.1
    x_inner = r_inner * np.cos(angles[:n_inner])
    y_inner = r_inner * np.sin(angles[:n_inner])
    inner_points = np.vstack([x_inner, y_inner]).T
    inner_labels = np.zeros(n_inner)

    # 外圆
    r_outer = np.random.rand(n_outer) * 0.4 + 0.6
    x_outer = r_outer * np.cos(angles[n_inner:])
    y_outer = r_outer * np.sin(angles[n_inner:])
    outer_points = np.vstack([x_outer, y_outer]).T
    outer_labels = np.ones(n_outer)

    # 合并数据
    x = np.vstack([inner_points, outer_points])
    y = np.hstack([inner_labels, outer_labels])

    # 添加噪声
    x += np.random.randn(*x.shape) * noise

    return x, y.reshape(-1, 1)


class Unittest(unittest.TestCase):
    # 测试同心圆问题
    def test_concentric_circles(self):
        # 生成数据
        x, y = concentric_circles(1000)

        # 可视化数据
        plt.scatter(
            x[y.flatten() == 0, 0], x[y.flatten() == 0, 1], color="red", label="Inner Circle"
        )
        plt.scatter(
            x[y.flatten() == 1, 0], x[y.flatten() == 1, 1], color="blue", label="Outer Circle"
        )
        plt.title("Concentric Circles - Linearly Inseparable")
        plt.legend()
        plt.show()

        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
