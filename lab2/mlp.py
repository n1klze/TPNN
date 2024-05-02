import pickle

import numpy as np
from tqdm import tqdm


class MLP:
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_hidden_layers = len(hidden_sizes)
        self.weights = []
        # Инициализация весов
        layers = [input_size] + hidden_sizes + [output_size]
        for i in np.arange(0, len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1])
            self.weights.append(w / np.sqrt(layers[i]))

    def __sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x: float) -> float:
        return x * (1 - x)

    def forward_propagation(self, x: float) -> float:
        self.activations = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.weights) - 1):
            net = self.activations[layer].dot(self.weights[layer])
            out = self.__sigmoid(net)
            self.activations.append(out)
        net = self.activations[-1].dot(self.weights[-1])
        self.activations.append(net)
        return self.activations[-1]

    def backward_propagation(self, x: float, y: float, learning_rate: float) -> None:
        deltas = [None] * (self.num_hidden_layers + 1)
        deltas[-1] = y - self.activations[-1]
        for i in range(self.num_hidden_layers, 0, -1):
            deltas[i - 1] = np.dot(
                deltas[i], self.weights[i].T
            ) * self.__sigmoid_derivative(self.activations[i])

        # Обновление весов и смещений
        for i in range(self.num_hidden_layers + 1):
            self.weights[i] += learning_rate * np.dot(self.activations[i].T, deltas[i])

    def train(self, X: list, y: list, epochs: int, learning_rate: float) -> None:
        for epoch in tqdm(range(epochs)):
            for x, target in zip(X, y):
                # output = self.forward_propagation(x)
                self.backward_propagation(x, target, learning_rate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward_propagation(X)


def save_weights(model: MLP, filename: str) -> None:
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load_weights(filename: str) -> MLP:
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model
