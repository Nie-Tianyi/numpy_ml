"""
Linear Regression Model
"""


class LinearRegressionModel:
    """
    Linear Regression Model
    """

    def __init__(self, records=False):
        self.weights = None
        self.bias = None
        if records:
            self.losses = []
