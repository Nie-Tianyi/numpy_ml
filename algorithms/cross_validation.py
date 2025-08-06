"""
交叉验证
"""

from algorithms.model_abstract import MachineLearningModel


def cross_validation(data, model: MachineLearningModel, seed=1, ratio=0.6):
    """
    交叉验证，将数据划分为 6、2、2三个部分，分别用于训练，验证和测试

    :param data: 用于训练的数据集，shape应该是(m,n)，m为数据长度，n为数据维度
    :param model: 模型，需要有`fit()`方法
    :param ratio: 用于训练的数据的比例，默认是0.6，剩下的数据一半用于验证，一半用于测试
    :param seed: 用于随机采样的随机数种子
    """
    pass


def k_fold_cross_validation(data, model, k=10, seed=1):
    pass
