import numpy as np
from typing import Dict


class Adam:
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, learning_rate: float = 0.01):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.state : Dict[str, Dict[str, np.ndarray]] = {}
        self.learning_rate = learning_rate

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        self.t += 1
        for key in params:
            if key not in self.state:
                self.state[key] = {
                    'm': np.zeros_like(params[key]),
                    'v': np.zeros_like(params[key])
                }
            
            # Update moments
            self.state[key]['m'] = self.beta1 * self.state[key]['m'] + (1 - self.beta1) * grads[key]
            self.state[key]['v'] = self.beta2 * self.state[key]['v'] + (1 - self.beta2) * (grads[key]**2)
            
            # Bias correction
            m_hat = self.state[key]['m'] / (1 - self.beta1**self.t)
            v_hat = self.state[key]['v'] / (1 - self.beta2**self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop:
    def __init__(self, beta2: float = 0.9, epsilon: float = 1e-8):
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state: Dict[str, np.ndarray] = {}

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray], learning_rate: float):
        for key in params:
            if key not in self.state:
                self.state[key] = np.zeros_like(params[key])
            
            # Update squared gradient cache
            self.state[key] = self.rho * self.state[key] + (1 - self.rho) * grads[key]**2
            
            # Update parameters
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.state[key]) + self.epsilon)



class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
