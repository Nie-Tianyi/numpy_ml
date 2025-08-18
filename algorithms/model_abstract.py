import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from algorithms.evaluation import EvaluationMethod
from algorithms.regularization import Regularization


class MachineLearningModel(ABC):
	def __init__(
		self,
		regularization: type[Regularization],
		niter=1000,
		learning_rate: float = 1,
		reg_param=0.03,
	):
		self.weights: Optional[NDArray[np.float64]] = None
		self.bias: Optional[NDArray[np.float64]] = None
		self.niter: int = niter
		self.lr: float = learning_rate
		self.lambda_: float = reg_param
		self.reg: type[Regularization] = regularization
		self.loss_history: List[float] = []

	@abstractmethod
	def fit(self, x, y):
		pass

	@abstractmethod
	def predict(self, x):
		pass

	@abstractmethod
	def evaluate(self, x_test, y_test, evaluation_method: type[EvaluationMethod]) -> float:
		"""
		用测试数据评估模型
		:param x_test: 测试输入数据
		:param y_test: 测试label
		:param evaluation_method: 可选参数，默认是None，可以是
		"""
		pass

	def plot_loss_history(self, title="Training Loss History", label="Loss") -> None:
		"""
		plot loss history
		"""
		seaborn.lineplot(self.loss_history)
		plt.title(title)
		plt.xlabel("Iteration")
		plt.ylabel(label)
		plt.show()  # 绘制图形

	def save(self, path: str) -> None:
		"""保存模型到文件（使用 pickle 格式）"""
		# 确保目录存在
		os.makedirs(os.path.dirname(path), exist_ok=True)

		with open(path, "wb") as f:
			pickle.dump(self, f)

	@staticmethod
	def load(path: str):
		"""从文件加载模型（使用 pickle 格式）"""
		with open(path, "rb") as f:
			return pickle.load(f)
