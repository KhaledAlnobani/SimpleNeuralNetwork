import numpy as np


def scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_scaled = (X - mean) / std
    return X_scaled


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true) * 100


def train_test_split(X, y, random_state=None, test_size=0.2):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.array(X)
    y = np.array(y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * test_size)
    X = X[indices, :]
    y = y[indices, :]
    X_train, y_train = X[split:], y[split:]
    X_test, y_test = X[:split], y[:split]
    return X_train, y_train, X_test, y_test


