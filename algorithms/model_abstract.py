from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from algorithms.regularization import Regularization


class MachineLearningModel(ABC):
    def __init__(
        self,
        niter=1000,
        learning_rate: float = 1,
        reg_param=0.03,
        regularization=Regularization.RIDGE,
    ):
        self.weights: Optional[NDArray[np.float64]] = None
        self.bias: Optional[NDArray[np.float64]] = None
        self.niter: int = niter
        self.lr: float = learning_rate
        self.lambda_: float = reg_param
        self.reg: Regularization = regularization
        self.loss_history: List[float] = []

    @abstractmethod
    def fit(self, x, y):
        pass

    def plot_loss_history(self, title="Training Loss History") -> None:
        """
        plot loss history
        """
        seaborn.lineplot(self.loss_history)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Cross-Entropy Loss")
        plt.show()
