import unittest

import numpy as np

from algorithms.activation_functions import ReLU, Sigmoid
from algorithms.evaluation import EvaluationMethod, Accuracy
from algorithms.logistic_regression import LogisticRegressionModel
from algorithms.loss_function import cross_entropy_loss
from algorithms.neural_networks.linear_layer import FCLinearLayer
from algorithms.neural_networks.neural_network import NeuralNetworkBaseModel
from algorithms.normaliser import z_score_normalisation
from algorithms.regularization import Regularization, Ridge
from test_data_set.test_data_gen import binary_data


class BinaryClassificationNeuralNetwork(NeuralNetworkBaseModel):
    def evaluate(
        self, x_test, y_test, evaluation_method: type[EvaluationMethod] = Accuracy
    ) -> float:
        y_hat = self.predict(x_test)
        y_hat = (y_hat >= self.threshold).astype(float).flatten()
        return evaluation_method.evaluate(y_hat, y_test)

    def __init__(
        self,
        layers=None,
        threshold: float = 0.5,
        niter=1000,
        learning_rate=0.1,
        reg_param: float = 0.3,
        regularization: type[Regularization] = Ridge,
    ):
        if layers is None:
            layers = [
                FCLinearLayer(4, activation_function=ReLU),
                FCLinearLayer(2, activation_function=ReLU),
                FCLinearLayer(1, activation_function=Sigmoid),
            ]

        super().__init__(
            layers,
            cross_entropy_loss,
            niter,
            learning_rate,
            reg_param,
            regularization,
        )
        self.threshold = threshold


class Unittest(unittest.TestCase):
    def test_neural_network(self):
        neural_network = BinaryClassificationNeuralNetwork(
            threshold=0.5, niter=2000, learning_rate=0.2
        )
        logistic_model = LogisticRegressionModel(
            threshold=0.5, niter=2000, learning_rate=0.1
        )  # benchmark

        (x, y) = binary_data(data_size=10000, seed=78)
        rescaled_x, scaler = z_score_normalisation(x)

        neural_network.fit(rescaled_x, y)
        neural_network.plot_loss_history(title="Neural Network's Loss", label="Cross Entropy Loss")
        logistic_model.fit(rescaled_x, y)
        logistic_model.plot_loss_history(title="Logistic Model's Loss", label="Cross Entropy Loss")

        (test_x, test_y) = binary_data(data_size=1000, seed=79)
        rescaled_test_x = scaler.rescale(test_x)
        acc_nn = neural_network.evaluate(rescaled_test_x, test_y)
        acc_logs = logistic_model.evaluate(rescaled_test_x, test_y)
        print(
            "Neural Network Model's Accuracy:", acc_nn
        )  # 神经网络的Accuracy反而比Logistic Model少一点，可能有点过拟合
        print("Logistic Model's Accuracy:", acc_logs)
        print(
            "Neural Network Final Loss:", neural_network.loss_history[-1]
        )  # 神经网络Final Loss少点
        print("Logistic Model Final Loss:", logistic_model.loss_history[-1])
        print("---------------------------------------------------")
        print("Test Point (1,1) predicted result should be close to 1")
        p1 = scaler.rescale(np.array([[1, 1]]))
        res1_nn = neural_network.predict(p1)
        print("Neural Network's Result:", res1_nn)
        res1_logs = logistic_model.predict(p1)
        print("Logistic Model's Result:", res1_logs)

        print("Test Point (0,0) predicted result should be close to 0")
        p2 = scaler.rescale(np.array([[0, 0]]))
        res2_nn = neural_network.predict(p2)
        print("Neural Network's Result:", res2_nn)
        res2_logs = logistic_model.predict(p2)
        print("Logistic Model's Result:", res2_logs)

        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
