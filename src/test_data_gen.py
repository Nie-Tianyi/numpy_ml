import numpy as np

def linear_data(data_size: int = 1000, seed=1):
    np.random.seed(seed)
    x_1 = np.random.rand(data_size)
    x_2 = np.random.rand(data_size)
    noise = np.random.randn(data_size)  # 创建1维噪声数组

    # 创建特征矩阵 (1000, 2)
    x = np.stack([x_1, x_2], axis=1)
    # 创建目标值 (1000,)
    y = 0.99 * x_1 + 2.3 * x_2 + noise + 1
    return x, y