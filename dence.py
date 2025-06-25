import numpy as np


class Dense:
    def __init__(self, input_size, num_neurons, activation, activation_prime):
        self.weights = np.random.randn(input_size, num_neurons) * 0.01
        self.bias = np.zeros((1, num_neurons))
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.z = None
        self.dw = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)
    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.bias        
        return self.activation(self.z)
    

    def backward(self, output_error, learning_rate):
        dz = self.activation_prime(self.z) * output_error
        self.dw = 1 / 80  * np.dot(self.input.T, dz) 
        self.db = 1 / 80 * np.sum(dz, axis=0, keepdims=True) 

        self.weights -= learning_rate * self.dw
        self.bias -= learning_rate * self.db
        return np.dot(dz, self.weights.T)
