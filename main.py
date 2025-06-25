import pandas as pd
from utils import scale, train_test_split, accuracy
from dence import Dense
from sequential_model import Sequential
from sklearn.datasets import load_iris
from activations import *
import matplotlib

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['target'] = iris.target
df = df[df['target'] != 2] # Select two classes for binary classification

y = df['target']
y = y.values.reshape(-1, 1)
df = df.drop('target', axis=1)

# scale data
scaled_data = scale(df)


X_train, y_train, X_test, y_test = train_test_split(scaled_data, y, test_size=0.2)


layers = [
    Dense(input_size=4, num_neurons=10, activation=ReLU, activation_prime=ReLU_prime),
    Dense(input_size=10, num_neurons=10, activation=ReLU, activation_prime=ReLU_prime),
    Dense(input_size=10, num_neurons=1, activation=sigmoid, activation_prime=sigmoid_prime)
]
model = Sequential(layers)
model.fit(X_train, y_train, epochs=200, learning_rate=1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
predictions = model.predict(X_test)
print(predictions[5:])  # Print first 5 predictions
print(y_test[5:])  # Print first 5 true labels

predictions = (predictions > 0.5).astype(int)

acc = accuracy(predictions, y_test)
print(f"Accuracy: {acc:.2f}%")
