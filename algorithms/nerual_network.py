import unittest
from typing import List

from tqdm import tqdm

from algorithms.loss_function import cross_entropy_loss
from algorithms.model_abstract import MachineLearningModel
from algorithms.neural_network_layer import NeuralNetworkLayer, LinearLayer, SigmoidOutputLayer
from algorithms.regularization import Regularization, Ridge


class NeuralNetwork(MachineLearningModel):
	"""
	神经网络
	"""

	def __init__(
		self,
		layers: List[NeuralNetworkLayer],
		loss_function,
		niter=1000,
		learning_rate=0.1,
		reg_param: float = 0.3,
		regularization: type[Regularization] = Ridge,
	):
		super().__init__(regularization, niter, learning_rate, reg_param)
		self.layers = layers
		self.loss_function = loss_function

	def __init_weights_and_bias(self, dim):
		# 逐层初始化权重和bias
		for layer in self.layers:
			layer.init_weights_and_bias(dim)
			dim = layer.num

	def fit(self, x, y):
		# 先初始化神经网络的参数
		(m, dim) = x.shape
		self.__init_weights_and_bias(dim)

		for _ in tqdm(range(self.niter)):
			y_hat = self.forward_propagation(x)
			self.backward_propagation(y_hat - y)
			self.loss_history.append(self.loss_function(y_hat, y))

	def predict(self, x):
		return self.forward_propagation(x)

	def forward_propagation(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def backward_propagation(self, error):
		for layer in reversed(self.layers):
			error = layer.backward(error)

	def evaluate(self, x_test, y_test) -> float:
		pass


class Unittest(unittest.TestCase):
	def test_neural_network(self):
		neural_network = NeuralNetwork(
			[LinearLayer(4), LinearLayer(3), SigmoidOutputLayer()], loss_function=cross_entropy_loss
		)

		self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
	unittest.main()
