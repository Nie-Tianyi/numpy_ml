import unittest

import numpy as np
from matplotlib import pyplot as plt


def spiral_data(n_samples=1000, n_classes=3, noise=0.1):
    """生成螺旋数据集"""
    x = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype="int")

    for j in range(n_classes):
        ix = range(n_samples * j, n_samples * (j + 1))
        r = np.linspace(0.0, 1, n_samples)  # 半径
        t = np.linspace(j * 4, (j + 1) * 4, n_samples) + np.random.randn(n_samples) * noise  # 角度
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return x, y.reshape(-1, 1)


class Unittest(unittest.TestCase):
    def test_spiral_problem(self):
        n_classes = 9
        # 生成数据
        x, y = spiral_data(300, n_classes)

        # 可视化数据
        colors = ["red", "blue", "green", "yellow", "grey", "purple", "black", "orange", "indigo"]
        for i in range(n_classes):
            plt.scatter(
                x[y.flatten() == i, 0], x[y.flatten() == i, 1], color=colors[i], label=f"Class {i}"
            )
        plt.title("Spiral Problem - Linearly Inseparable")
        plt.legend()
        plt.show()

        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
