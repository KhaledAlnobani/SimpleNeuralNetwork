import numpy as np
from typing import Dict,Callable


class Dense:
    def __init__(self, input_size: int, num_neurons :int , activation :Callable , activation_prime: Callable):

        scale = np.sqrt(2. / input_size) if activation.__name__ == "relu" else np.sqrt(1. / input_size) 
        self.weights = np.random.randn(input_size, num_neurons) * scale
        self.bias = np.zeros((1, num_neurons)) 
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.z = None
        
    def forward(self, input_data: np.ndarray)-> np.ndarray:
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.bias        
        output = self.activation(self.z)
        return output
    

    def backward(self, d_out: np.ndarray, batch_size: int)-> Dict[str, np.ndarray]:
        dz = self.activation_prime(self.z) * d_out
        dw = 1 / batch_size  * np.dot(self.input.T, dz) 
        db = 1 / batch_size * np.sum(dz, axis=0, keepdims=True)
        dx = np.dot(dz, self.weights.T) 

        return {'dw' : dw, 'db' : db, 'dx' : dx}
