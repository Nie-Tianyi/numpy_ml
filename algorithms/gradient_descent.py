import numpy as np
from numpy.typing import NDArray


def compute_gradient(
	x: NDArray[np.float64], y_pred: NDArray[np.float64], y_real: NDArray[np.float64]
):
	"""
	计算梯度
	:param x: 输入数据，shape为(m,n)，m为样本数量，n为特征数量
	:param y_pred: 模型预测值，shape为(m,1)，多分类问题则为(m,K)
	:param y_real: 真实值，shape为(m,1)，多分类问题则为(m,K)
	:return: 梯度，dlt_w的shape为(n,1), dlt_b的shape为(1,1)；多分类问题则为(n,K)
	"""
	m = x.shape[0]
	dlt_w = (1 / m) * np.dot(x.T, (y_pred - y_real))
	dlt_b = (1 / m) * np.sum(y_pred - y_real)
	return dlt_w, dlt_b
