from typing import List


from algorithms.activation_functions import LinearUnit
from algorithms.evaluation import EvaluationMethod, MeanSquaredError
from algorithms.neural_networks.linear_layer import LinearLayer
from algorithms.neural_networks.neural_network import NeuralNetwork
from algorithms.neural_networks.neural_network_layer_abstract import NeuralNetworkLayer
from algorithms.regularization import Regularization, Ridge


class LinearNeuralNetwork(NeuralNetwork):
    def evaluate(
        self, x_test, y_test, evaluation_method: type[EvaluationMethod] = MeanSquaredError
    ) -> float:
        pass

    def __init__(self, layers: List[NeuralNetworkLayer] = None, niter=1000):
        if layers is None:
            layers = [
                LinearLayer(4, activation_function=LinearUnit),
                LinearLayer(2, activation_function=LinearUnit),
                LinearLayer(1, activation_function=LinearUnit),
            ]
        super().__init__(
            layers, MeanSquaredError, niter, learning_rate=0.1, reg_param=0.3, regularization=Ridge
        )
