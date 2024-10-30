from loss_functions import *


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.errors = []

    def fit(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward Propagation
            output = X
            for layer in self.layers:  # or output = predict(output)
                output = layer.forward(output)

            error = binary_cross_entropy(output, y)
            self.errors.append(error)

            # Backward Propagation

            loss_gradient = binary_cross_entropy_prime(output, y)

            for layer in reversed(self.layers):
                loss_gradient = layer.backward(loss_gradient, learning_rate)

            if epoch % 100 == 0:
                print(f'Error at epoch {epoch}: {error}')

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
