import numpy as np


class Dense:
    def __init__(self, input_size, num_neurons, activation, activation_prime):
        self.weights = np.random.randn(input_size, num_neurons) * 0.01
        self.bias = np.zeros((1, num_neurons))
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.z = None

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.bias
        return self.activation(self.z)

    def backward(self, output_error, learning_rate):
        dz = self.activation_prime(self.z) * output_error
        dw = np.dot(self.input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        return np.dot(dz, self.weights.T)
