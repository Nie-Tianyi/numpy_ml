import numpy as np
import numba


@numba.njit
def mean_square_error(y_predict, y_real):
    r"""
    y_predict & y should be the same shape
    :param y_predict: predicted value, a scalar or a numpy array
    :param y_real: real label value, a scalar or a numpy array
    :return: MSE loss
    """
    return 0.5 * np.mean((y_predict ** 2 - y_real ** 2))


if __name__ == "__main__":
    mse = mean_square_error(np.array([1,2,3,4]), np.array([11,4,2,1]))
    print(mse)