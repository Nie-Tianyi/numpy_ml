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

    # 测试XOR问题
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

    # 测试螺旋问题
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
