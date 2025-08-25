import unittest

import numpy as np
from matplotlib import pyplot as plt


def xor_data(size=1000, noise=0.1):
    """生成XOR问题数据集"""
    x = np.random.randn(size, 2)
    y = np.zeros(size)

    # 定义XOR逻辑
    for i in range(size):
        if (x[i, 0] > 0 > x[i, 1]) or (x[i, 0] < 0 < x[i, 1]):
            y[i] = 1

    # 添加一些噪声
    noise_mask = np.random.rand(size) < noise
    y[noise_mask] = 1 - y[noise_mask]

    return x, y.reshape(-1, 1)


class Unittest(unittest.TestCase):
    def test_xor_problem(self):
        # 生成数据
        x, y = xor_data(1000)

        # 可视化数据
        plt.scatter(x[y.flatten() == 0, 0], x[y.flatten() == 0, 1], color="red", label="Class 0")
        plt.scatter(x[y.flatten() == 1, 0], x[y.flatten() == 1, 1], color="blue", label="Class 1")
        plt.title("XOR Problem - Linearly Inseparable")
        plt.legend()
        plt.show()

        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
