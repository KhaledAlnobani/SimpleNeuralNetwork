import numpy as np
def binary_cross_entropy(y_pred, y_true,epsilon = 1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_pred, y_true,epsilon = 1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred + (1 - y_true) / (1 - y_pred)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
