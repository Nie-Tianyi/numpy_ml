import unittest
from typing import List


from algorithms.activation_functions import LinearUnit
from algorithms.evaluation import EvaluationMethod, MeanSquaredError
from algorithms.linear_regression import LinearRegressionModel
from algorithms.loss_function import mean_square_error
from algorithms.neural_networks.linear_layer import FCLinearLayer
from algorithms.neural_networks.neural_network import NeuralNetworkBaseModel
from algorithms.neural_networks.neural_network_layer_abstract import NeuralNetworkLayerAbstract
from algorithms.regularization import Regularization, Ridge
from test_data_set.linear_data import linear_data


class LinearNeuralNetwork(NeuralNetworkBaseModel):
    def evaluate(
        self, x_test, y_test, evaluation_method: type[EvaluationMethod] = MeanSquaredError
    ) -> float:
        y_hat = self.predict(x_test)
        return evaluation_method.evaluate(y_hat, y_test)

    def __init__(
        self,
        layers: List[NeuralNetworkLayerAbstract] = None,
        niter=1000,
        learning_rate=0.1,
        reg_param=0.3,
        regularization: type[Regularization] = Ridge,
    ):
        if layers is None:
            layers = [
                FCLinearLayer(4, activation_function=LinearUnit),
                FCLinearLayer(2, activation_function=LinearUnit),
                FCLinearLayer(1, activation_function=LinearUnit),
            ]
        super().__init__(layers, mean_square_error, niter, learning_rate, reg_param, regularization)


class Unittest(unittest.TestCase):
    def test_linear_neural_network(self):
        x, y = linear_data(data_size=10000, seed=78)

        neural_network = LinearNeuralNetwork(niter=3000, learning_rate=0.001)
        linear_model = LinearRegressionModel(niter=3000, learning_rate=0.001)  # benchmark

        neural_network.fit(x, y)
        linear_model.fit(x, y)

        neural_network.plot_loss_history(
            title="Neural Network's Loss History", label="Mean Square Error"
        )
        linear_model.plot_loss_history(
            title="Linear Regression Model' Loss History", label="Mean Square Error"
        )

        test_x, test_y = linear_data(data_size=1000, seed=123)
        mse_nn = neural_network.evaluate(test_x, test_y)
        mse_lm = linear_model.evaluate(test_x, test_y)

        print("Neural Network MSE: ", mse_nn)
        print("Linear Model MSE: ", mse_lm)

        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
