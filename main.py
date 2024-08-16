import pandas as pd
import numpy as np


# Activation Functions
def ReLU(x):
    return np.maximum(0, x)


def ReLU_prime(x):
    return np.array(x > 0, dtype=float)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Loss Functions
def cross_entropy_loss(predictions, targets):
    m = predictions.shape[1]
    return (1 / m) * -np.sum(targets * np.log(predictions + 1e-10))


# Utility Functions
def train_test_split(X, y, random_state=None, test_size=0.2):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * test_size)
    X_train, y_train = X[split:], y[split:]
    X_test, y_test = X[:split], y[:split]
    return X_train, y_train, X_test, y_test


def one_hot_encode(y):
    return pd.get_dummies(y).values


# Dense Layer
class Dense:
    def __init__(self, input_size, num_neurons, activation, activation_prime):
        self.weights = np.random.randn(input_size, num_neurons)
        self.bias = np.zeros((1, num_neurons))
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(np.dot(self.input, self.weights) + self.bias)
        return self.output

    def backward(self, output_error, learning_rate, flag=False):
        weight_error = np.dot(self.input.T, output_error) / output_error.shape[0]

        self.weights -= learning_rate * weight_error
        self.bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        # print(1)
        if not flag:
            input_error = np.dot(output_error, self.weights.T) * self.activation_prime(self.input)
        else:
            input_error = np.dot(output_error, self.weights.T)
        return input_error / output_error.shape[0]


# Neural Network
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

            # Backward Propagation
            error = output - y

            flag = True
            for layer in reversed(self.layers):
                error = layer.backward(error, learning_rate, flag)
                flag = False

            # Print Error for Monitoring
            if epoch % 100 == 0:
                loss = cross_entropy_loss(output, y)
                self.errors.append(loss)
                print(f'Error at epoch {epoch}: {loss}')

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output


# data
df = pd.read_csv('mnist_test.csv')
X = df.drop(['label'], axis=1).values / 255.0
y = df['label']

X = np.array(X)
y = np.array(y)
y = one_hot_encode(y)
X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=48, test_size=0.2)

# Define and Train the Model

np.random.seed(10)
layers = [
    Dense(input_size=784, num_neurons=25, activation=ReLU, activation_prime=ReLU_prime),
    Dense(input_size=25, num_neurons=15, activation=ReLU, activation_prime=ReLU_prime),
    Dense(input_size=15, num_neurons=10, activation=softmax, activation_prime=None)
]

model = Sequential(layers)
model.fit(X_train, y_train, epochs=10000, learning_rate=0.002)
