import copy
from loss_functions import *
import numpy as np


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.errors = []

    def gradient_check(self, X, y, epsilon=1e-7):
        """
        Check the gradients using numerical approximation.
        """

        output = X
        for layer in self.layers:
            output = layer.forward(output)

        # Step 2: Backward on original model (do not update weights)
        loss_gradient = binary_cross_entropy_prime(output, y)
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate=0) 
        copied_layers = copy.deepcopy(self.layers)
        gradapprox = []

        for layer_index, layer in enumerate(copied_layers):
            weights_shape = layer.weights.shape
            flat_weights = layer.weights.flatten()

            for i in range(len(flat_weights)):
                # + epsilon
                weights_plus = np.copy(flat_weights)
                weights_plus[i] += epsilon
                copied_layers[layer_index].weights = weights_plus.reshape(weights_shape)

                output = X
                for l in copied_layers:
                    output = l.forward(output)
                J_plus = binary_cross_entropy(output, y)

                # - epsilon
                weights_minus = np.copy(flat_weights)
                weights_minus[i] -= epsilon
                copied_layers[layer_index].weights = weights_minus.reshape(weights_shape)

                output = X
                for l in copied_layers:
                    output = l.forward(output)
                J_minus = binary_cross_entropy(output, y)

                # numerical gradient
                grad = (J_plus - J_minus) / (2 * epsilon)
                gradapprox.append(grad)

            # Reset weights for this layer to original
            # copied_layers[layer_index].weights = flat_weights.reshape(weights_shape)


            bias_shape = layer.bias.shape
            flat_bias = layer.bias.flatten()

            for i in range(len(flat_bias)):
                # + epsilon
                bias_plus = np.copy(flat_bias)
                bias_plus[i] += epsilon
                copied_layers[layer_index].bias = bias_plus.reshape(bias_shape)

                output = X
                for l in copied_layers:
                    output = l.forward(output)
                J_plus = binary_cross_entropy(output, y)

                # - epsilon
                bias_minus = np.copy(flat_bias)
                bias_minus[i] -= epsilon
                copied_layers[layer_index].bias = bias_minus.reshape(bias_shape)

                output = X
                for l in copied_layers:
                    output = l.forward(output)
                J_minus = binary_cross_entropy(output, y)

                # numerical gradient for bias[i]
                grad = (J_plus - J_minus) / (2 * epsilon)
                gradapprox.append(grad)

            # After backpropagation
        analytical_grads = []  # Flatten all layer.dw and layer.db

        for layer in self.layers:
            analytical_grads.extend(layer.dw.flatten())
            analytical_grads.extend(layer.db.flatten())

        analytical_grads = np.array(analytical_grads)

        # Now compare
        numerator = np.linalg.norm(analytical_grads - gradapprox)
        denominator = np.linalg.norm(analytical_grads) + np.linalg.norm(gradapprox)
        difference = numerator / denominator

        print("Gradient Check Difference:", difference)
        if difference < 1e-7:
            print(f"Gradient check passed!, difference {difference}")
        else:
            print(f"Gradient check failed!, difference is too high: {difference}")
                    
                    
    
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

            if epoch % 10 == 0:
                print(f'Error at epoch {epoch}: {error}')

            if epoch % 50 == 0:
                self.gradient_check(X, y)


                                

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
