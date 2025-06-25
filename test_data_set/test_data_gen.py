"""
生成模型的测试数据
"""

import numpy as np


def linear_data(data_size: int = 1000, seed=1):
    """
    生成线性数据，0.99 * x_1 + 2.3 * x_2 + 1 + random_noise
    noise符合正态分布
    :param data_size: 生成多少条数据
    :param seed: 随机数生成的种子
    :return: [x_1, x_2], y
    """
    np.random.seed(seed)
    x_1 = np.random.rand(data_size)
    x_2 = np.random.rand(data_size)
    noise = np.random.randn(data_size)  # 创建1维噪声数组

    # 创建特征矩阵 (1000, 2)
    x = np.stack([x_1, x_2], axis=1)
    # 创建目标值 (1000,)
    y = 0.99 * x_1 + 2.3 * x_2 + noise + 1
    return x, y
