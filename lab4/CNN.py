import numpy as np
from scipy import signal


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def categorical_cross_entropy(y, y_pred):
    ce_loss = np.multiply(y, np.log(y_pred))
    return -np.sum(ce_loss)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    x = np.array(x)
    exp_values = np.exp(x)
    exp_sum = np.sum(exp_values)
    softmax_values = exp_values / exp_sum

    return softmax_values


def softmax_derivative(x):
    x = np.array(x)
    softmax_values = softmax(x)
    softmax_derivative_values = softmax_values * (1 - softmax_values)

    return softmax_derivative_values


def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))


def categorical_cross_entropy_derivative(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return -(y_true / y_pred)


def binary_cross_entropy_derivative(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - (y_true / y_pred)) / np.size(y_true)


class Lay:
    def __init__(self, activation=None):
        self.input = None
        self.output = None
        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "softmax":
            self.activation = Softmax()
        else:
            self.activation = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class CNN:
    def __init__(self, loss, loss_derivative, learning_rate, epochs):
        self.layers = []
        self.loss = loss
        self.loss_derivative = loss_derivative
        self.learning_rate = learning_rate
        self.epochs = epochs

    def add_lay(self, lay: Lay):
        self.layers.append(lay)
        if lay.activation is not None:
            self.layers.append(lay.activation)
        return self

    def predict(self, input):
        output = input
        for lay in self.layers:
            output = lay.forward(output)
        return output

    def test(self, x_test, y_test):
        predictions = []
        for x, y in zip(x_test, y_test):
            output = self.predict(x)
            predictions.append(output)

        predictions = np.array(predictions)
        return predictions

    def train(self, x_train, y_train, verbose=True):
        for e in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.predict(x)

                grad = self.loss_derivative(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, self.learning_rate)

            if verbose:
                print(f"{e + 1}/{self.epochs}")


class Dense(Lay):
    def __init__(self, input_size, output_size, activation):
        super().__init__(activation)
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = np.array(input, dtype=np.float64)
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        output_gradient = np.array(output_gradient, dtype=np.float64)
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Reshape(Lay):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)


class Convolutional(Lay):
    def __init__(self, input_shape, kernel_size, kernels, activation):
        super().__init__(activation)
        input_depth, input_height, input_width = input_shape
        self.depth = kernels
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (
            kernels,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )
        self.kernels_shape = (kernels, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid"
                )
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], "valid"
                )
                input_gradient[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], "full"
                )

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient


class MaxPooling(Lay):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input
        depth, height, width = input.shape
        pooled_height = height // self.pool_size
        pooled_width = width // self.pool_size
        self.output = np.zeros((depth, pooled_height, pooled_width))

        for d in range(depth):
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    h_start = ph * self.pool_size
                    h_end = h_start + self.pool_size
                    w_start = pw * self.pool_size
                    w_end = w_start + self.pool_size
                    window = input[d, h_start:h_end, w_start:w_end]
                    self.output[d, ph, pw] = np.max(window)

        return self.output

    def backward(self, output_gradient, learning_rate):
        depth, pooled_height, pooled_width = output_gradient.shape
        input_gradient = np.zeros_like(self.input)

        for d in range(depth):
            for ph in range(pooled_height):
                for pw in range(pooled_width):
                    h_start = ph * self.pool_size
                    h_end = h_start + self.pool_size
                    w_start = pw * self.pool_size
                    w_end = w_start + self.pool_size
                    window = self.input[d, h_start:h_end, w_start:w_end]
                    max_value = np.max(window)
                    mask = window == max_value
                    input_gradient[d, h_start:h_end, w_start:w_end] += (
                        mask * output_gradient[d, ph, pw]
                    )

        return input_gradient


class Activation(Lay):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_derivative)


class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, softmax_derivative)
