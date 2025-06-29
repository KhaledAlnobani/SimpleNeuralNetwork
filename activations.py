import numpy as np
def ReLU(x):
    return np.maximum(0, x)

def ReLU_prime(x):
    return (x > 0).astype(np.float32)
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z + 1e-10))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)
