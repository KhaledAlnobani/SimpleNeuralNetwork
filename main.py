import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



# Activation Functions
def ReLU(x):
    return np.maximum(0, x)


def ReLU_prime(x):
    return np.where(x > 0, 1, 0)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)
def binary_cross_entropy(y_pred,y_true):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_pred, y_true):
    return ((1 - y_true) / (y_pred) - y_true / (y_pred)) / np.size(y_true)


# Utility Functions
# def train_test_split(X, y, random_state=None, test_size=0.2):
#     if random_state is not None:
#         np.random.seed(random_state)
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)
#     split = int(len(X) * test_size)
#     X_train, y_train = X[split:], y[split:]
#     X_test, y_test = X[:split], y[:split]
#     return X_train, y_train, X_test, y_test
#


# Dense Layer
class Dense:
    def __init__(self, input_size, num_neurons, activation, activation_prime):
        self.weights = np.random.randn(input_size, num_neurons)* 0.1
        self.bias = np.zeros((1,num_neurons))
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.z = None

    def forward(self, input_data):
        self.input = input_data
        self.z = self.activation(np.dot(self.input, self.weights) + self.bias)
        return self.z

    def backward(self, output_error, learning_rate):
        dz = self.activation_prime(self.z)* output_error
        dw = np.dot(self.input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        # print(np.dot(dz,self.weights.T).shape)
        return np.dot(dz,self.weights.T)


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

            dl = binary_cross_entropy_prime(output, y)

            for layer in reversed(self.layers):
                dl = layer.backward(dl, learning_rate)

            # Print Error for Monitoring
            if epoch % 10 == 0:
                loss = binary_cross_entropy(output, y)
                self.errors.append(loss)
                print(f'Error at epoch {epoch}: {loss}')

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output



df = pd.read_excel("Raisin_Dataset.xlsx")


df=pd.get_dummies(df, columns=['Class'], drop_first=True)

target = df["Class_Kecimen"].astype(int)
target = target.values.reshape(-1,1)
df= df.drop("Class_Kecimen", axis=1)

scaler = StandardScaler()
data = scaler.fit_transform(df)

# iris = load_iris()

# Create a DataFrame with the feature data
# df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target column
# df['species'] = iris.target

# One-hot encoding for the target (optional)
# df = pd.get_dummies(df, columns=['species'], drop_first=True)

# Preview the dataset

# data
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=4)

# Define and Train the Model

# np.random.seed(10)
layers = [
    Dense(input_size=7, num_neurons=5, activation=ReLU, activation_prime=ReLU_prime),
    Dense(input_size=5, num_neurons=3, activation=ReLU, activation_prime=ReLU_prime),
    Dense(input_size=3, num_neurons=1, activation=sigmoid, activation_prime=sigmoid_prime)
]
model = Sequential(layers)
model.fit(X_train, y_train, epochs=420, learning_rate=0.02)

predictions = model.predict(X_test)

# Convert predictions to binary values (0 or 1)
predictions_binary = (predictions > 0.5).astype(int)


# Calculate accuracy
accuracy = np.mean(predictions_binary == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
