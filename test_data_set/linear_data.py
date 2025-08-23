"""
生成模型的测试数据
"""

import unittest

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def linear_data(data_size: int = 1000, seed=1):
    """
    生成线性数据，0.99 * x_1 + 2.3 * x_2 + 1 + random_noise
    noise符合正态分布
    :param data_size: 生成多少条数据
    :param seed: 随机数生成的种子
    :return: [x_1, x_2], [y]
    """
    np.random.seed(seed)
    x_1 = np.random.rand(data_size)
    x_2 = np.random.rand(data_size)
    noise = np.random.randn(data_size)  # 创建1维噪声数组

    # 创建特征矩阵 (1000, 2)
    x = np.stack([x_1, x_2], axis=1)
    # 创建目标值 (1000,1)
    y = 0.99 * x_1 + 2.3 * x_2 + noise + 1  # y.shape = (1000,)
    y = y.reshape(-1, 1)  # y.shape = (1000,1)

    return x, y


def binary_data(data_size=1000, seed=1):
    """
    binary data generator
    """
    np.random.seed(seed)

    x_1 = np.random.rand(data_size)
    x_2 = np.random.rand(data_size)
    noise = np.random.rand(data_size)

    x = np.stack([x_1, x_2], axis=1)
    y = np.where(x_1 + x_2 + noise <= 1.5, 0, 1)  # y.shape = (1000,)
    y = y.reshape(-1, 1)  # y.shape = (1000,1)

    return x, y


class Unittest(unittest.TestCase):
    def test_linear_data(self):
        x, y = linear_data(data_size=1000, seed=1)

        # 转换为DataFrame便于Seaborn绘图
        df = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y[:, 0]})

        # 创建1x2的子图布局
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Linear Data Visualization (n={1000})", fontsize=16)

        # 绘制x1与y的关系
        sns.scatterplot(
            data=df,
            x="x1",
            y="y",
            alpha=0.3,  # 设置透明度避免重叠点
            ax=axes[0],
        )
        axes[0].set_title("Relationship between x1 and y")
        axes[0].set_xlabel("Feature x1")
        axes[0].set_ylabel("Target y")

        # 添加回归线
        sns.regplot(
            data=df,
            x="x1",
            y="y",
            scatter=False,  # 不显示散点（上面已显示）
            color="red",
            line_kws={"linewidth": 2},
            ax=axes[0],
        )

        # 绘制x2与y的关系
        sns.scatterplot(data=df, x="x2", y="y", alpha=0.3, ax=axes[1])
        axes[1].set_title("Relationship between x2 and y")
        axes[1].set_xlabel("Feature x2")
        axes[1].set_ylabel("Target y")

        # 添加回归线
        sns.regplot(
            data=df,
            x="x2",
            y="y",
            scatter=False,
            color="red",
            line_kws={"linewidth": 2},
            ax=axes[1],
        )

        # 添加3D散点图展示x1、x2与y的关系
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection="3d")

        # 抽样部分点避免3D图过于密集
        sample = df.sample(1000, random_state=1)

        sc = ax_3d.scatter(
            sample["x1"],
            sample["x2"],
            sample["y"],
            c=sample["y"],  # 根据y值着色
            cmap="viridis",
            alpha=0.7,
        )

        ax_3d.set_title("3D View: x1, x2 and y")
        ax_3d.set_xlabel("Feature x1")
        ax_3d.set_ylabel("Feature x2")
        ax_3d.set_zlabel("Target y")
        fig_3d.colorbar(sc, label="y value")

        # 显示所有图像
        plt.tight_layout()
        plt.show()

        self.assertEqual(1 + 1, 2)

    def test_binary_data(self):
        x, y = binary_data(data_size=1000, seed=777)
        # 可视化
        plt.figure(figsize=(10, 8))

        # 绘制数据点，按类别着色
        plt.scatter(
            x[y[:, 0] == 0, 0],
            x[y[:, 0] == 0, 1],
            color="blue",
            alpha=0.7,
            label="Class 0 (x1+x2+noise <= 1)",
        )
        plt.scatter(
            x[y[:, 0] == 1, 0],
            x[y[:, 0] == 1, 1],
            color="red",
            alpha=0.7,
            label="Class 1 (x1+x2+noise > 1)",
        )

        # 绘制理论决策边界 (x1 + x2 = 1.5)
        boundary_x = np.linspace(0, 1, 100)
        boundary_y = 1 - boundary_x
        plt.plot(
            boundary_x,
            boundary_y,
            "k--",
            linewidth=2,
            label="Theoretical Boundary (x1 + x2 = 1)",
        )

        # 美化图表
        plt.title("Binary Classification Data Visualization", fontsize=14)
        plt.xlabel("Feature x1", fontsize=12)
        plt.ylabel("Feature x2", fontsize=12)
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)

        # 添加噪声影响说明
        plt.text(
            0.05,
            0.95,
            "Noise effect: Points may cross boundary",
            fontsize=10,
            backgroundcolor="white",
        )

        plt.show()

        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
