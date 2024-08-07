import numpy as np


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Finds the mean squared error (MSE) between true and predicted values
    :param y_true: true values
    :param y_pred: predicted values
    :return: mean squared error (MSE)
    """
    return float(np.mean((y_true - y_pred) ** 2))


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Finds the root mean squared error (RMSE) between true and predicted values
    :param y_true: true values
    :param y_pred: predicted values
    :return: root mean squared error (RMSE)
    """
    return float(np.sqrt(MSE(y_true, y_pred)))


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Finds the mean absolute error (MAE) between true and predicted values
    :param y_true: true values
    :param y_pred: predicted values
    :return: mean absolute error (MAE)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Finds the mean absolute percentage error (MAPE) between true and predicted values
    :param y_true: true values
    :param y_pred: predicted values
    :return: mean absolute percentage error (MAPE)
    """
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))
