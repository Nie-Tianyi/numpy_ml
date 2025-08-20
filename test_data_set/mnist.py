import os.path
from typing import Union

import numpy as np
from numpy.random import Generator


def mnist(data_size: int = 1000, seed=1):
    """
    MNIST数据集，数据是一个28 * 28的灰度图像，每个像素值从 0~255，标签是一个0~9的数字
    返回mnist数据集，根据随机数种子在70000条样本里（无放回）随机取样
    :param data_size: 取多少条数据，最小为1，最大为70000
    :param seed: 随机取样的随机数种子
    :return: 返回一对 (x,y), x是 28 * 28 的图像， y是0~9的数字
    """

    assert 0 < data_size <= 70000, "Incompatible data size, data size should be in (0, 70000]"

    data_path: str = os.path.join(os.path.dirname(__file__), "./data/mnist.npz")
    data = np.load(data_path)
    x = data["images"]
    y = data["labels"]

    # 使用随机数生成器（确保可重复性）
    rng: Union[Generator, Generator] = np.random.default_rng(seed=seed)

    # 无放回随机采样索引
    indices = rng.choice(len(x), size=data_size, replace=False)

    return x[indices], y[indices].reshape(-1, 1)
