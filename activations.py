import numpy as np
def ReLU(x):
    return np.maximum(0, x)

def ReLU_prime(x):
    return np.where(x > 0, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z + 1e-10))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)
