import copy
from loss_functions import *
import numpy as np
from dense import Dense
from optimizers import Adam, RMSprop, SGD
from typing import Dict
from optimizers import Adam, RMSprop, SGD

class Sequential:
    def __init__(self, layers: list[Dense], optimizer=  None):
        self.optimizer = optimizer if optimizer else Adam()
        self.layers = layers
        self.history =  {'loss': []}

    def set_optimizer(self, optimizer_type: str, **kwargs):
        optimizers = {
            'adam': Adam,
            'rmsprop': RMSprop,
            'sgd': SGD
        }

        opt_cls = optimizers.get(optimizer_type.lower())
        if not opt_cls:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        self.optimizer = opt_cls(**kwargs)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        # Forward pass through all layers
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    
    def _backward(self, y_pred: np.ndarray, y_true: np.ndarray, batch_size : int) -> Dict[str, np.ndarray]:
        # Backward pass: compute gradients for all layers
        d_out = binary_cross_entropy_prime(y_pred, y_true)

        grads = {}
        num_layers = len(self.layers)
        for i, layer in enumerate(reversed(self.layers)):
            layer_index = num_layers - 1 - i
            layer_grads = layer.backward(d_out, batch_size)
            grads[f'layer_{layer_index}_w'] = layer_grads['dw']
            grads[f'layer_{layer_index}_b'] = layer_grads['db']
            d_out = layer_grads['dx']  # Pass gradient to previous layer
        return grads
    
    def _update_params(self, grads: Dict[str, np.ndarray]):
        # Update model parameters using the optimizer
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'layer_{i}_w'] = layer.weights
            params[f'layer_{i}_b'] = layer.bias

        self.optimizer.update(params, grads)
        
        # Update layer parameters
        for i, layer in enumerate(self.layers):
            layer.weights = params[f'layer_{i}_w']
            layer.bias = params[f'layer_{i}_b']
            


    def _gradient_check(self, X, y, epsilon=1e-7):
        """
        Performs gradient checking to compare analytical and numerical gradients.
        """
        # Forward pass
        y_pred = self._forward(X)

        #Analytical gradients
        analytical_grads = self._backward(y_pred, y, batch_size=X.shape[0])
        analytical = []
        for i, layer in enumerate(self.layers):
            analytical.extend(analytical_grads[f'layer_{i}_w'].flatten())
            analytical.extend(analytical_grads[f'layer_{i}_b'].flatten())
        analytical = np.array(analytical)

        #Numerical gradients
        gradapprox = []

        for layer_index, layer in enumerate(self.layers):
            # ----- Weights -----
            for idx in np.ndindex(layer.weights.shape):
                # + epsilon
                layer.weights[idx] += epsilon
                J_plus = binary_cross_entropy(self._forward(X), y)

                # - epsilon
                layer.weights[idx] -= 2 * epsilon
                J_minus = binary_cross_entropy(self._forward(X), y)

                # Restore
                layer.weights[idx] += epsilon

                grad = (J_plus - J_minus) / (2 * epsilon)
                gradapprox.append(grad)

            # ----- Biases -----
            for idx in np.ndindex(layer.bias.shape):
                # + epsilon
                layer.bias[idx] += epsilon
                J_plus = binary_cross_entropy(self._forward(X), y)

                # - epsilon
                layer.bias[idx] -= 2 * epsilon
                J_minus = binary_cross_entropy(self._forward(X), y)

                # Restore
                layer.bias[idx] += epsilon

                grad = (J_plus - J_minus) / (2 * epsilon)
                gradapprox.append(grad)


        gradapprox = np.array(gradapprox)

        # Compute relative difference between analytical and numerical gradients
        numerator = np.linalg.norm(analytical - gradapprox)
        denominator = np.linalg.norm(analytical) + np.linalg.norm(gradapprox) + 1e-10
        difference = numerator / denominator

        print("Gradient Check Difference:", difference)
        if difference < 1e-7:
            print(f"Gradient check passed! Difference: {difference:.2e}")
        else:
            print(f"Gradient check failed! Difference: {difference:.2e}")


    def fit(self, X, y, epochs, batch_size= None):
        m = X.shape[0]
        batch_size = batch_size if batch_size else m  # If batch_size is None, use the entire dataset as one batch

        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            for i in range(0, len(X), batch_size if batch_size else len(X)):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward Propagation
                output = self._forward(X_batch)
                # Backward Propagation
                grads = self._backward(output, y_batch, batch_size)

                # update parameters
                self._update_params(grads)

                # Compute and store loss
                loss = binary_cross_entropy(output, y_batch)
                self.history['loss'].append(loss)

            if epoch % 30 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
            if epoch % 50 == 0:
                self._gradient_check(X, y)


                                

    def predict(self, X:np.ndarray) ->np.ndarray:
        # Predict output for given input X
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
