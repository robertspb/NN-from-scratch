import numpy as np
import pandas as pd
import os
import requests
from matplotlib import pyplot as plt

ALPHA = 0.5
TARGETS = 10
BATCH_SIZE = 100
HIDDEN_LAYER_SIZE = 64


class TwoLayerNeural:
    def __init__(self, n_features, n_classes):
        self.input_1 = None
        self.output_1 = None
        self.input = None
        self.output = None
        self.X = None
        self.weights_1 = xavier(n_features, HIDDEN_LAYER_SIZE)
        self.biases_1 = xavier(1, HIDDEN_LAYER_SIZE)
        self.weights_2 = xavier(HIDDEN_LAYER_SIZE, n_classes)
        self.biases_2 = xavier(1, n_classes)

    def forward(self, X):
        self.X = X
        self.input_1 = np.dot(self.X, self.weights_1) + self.biases_1
        self.output_1 = sigmoid(self.input_1)
        self.input = np.dot(self.output_1, self.weights_2) + self.biases_2
        self.output = sigmoid(self.input)
        return self.output

    def backprop(self, X, y, alpha):
        error = mse_derivative(self.output, y)

        output_error = sigmoid_derivative(self.input) * error
        input_error = np.dot(output_error, self.weights_2.T)
        weights_error_2 = np.dot(self.output_1.T, output_error)
        self.weights_2 -= weights_error_2 * alpha
        self.biases_2 -= output_error.mean() * alpha

        output_error_1 = sigmoid_derivative(self.input_1) * input_error
        weights_error_1 = np.dot(self.X.T, output_error_1)
        self.weights_1 -= weights_error_1 * alpha
        self.biases_1 -= output_error_1.mean() * alpha


class OneLayerNeural:

    def __init__(self, n_features, n_classes):
        self.input = None
        self.output = None
        self.X = None
        self.weights = xavier(n_features, n_classes)
        self.biases = xavier(1, n_classes)

    def forward(self, X):
        self.X = X
        self.input = np.dot(self.X, self.weights) + self.biases
        self.output = sigmoid(self.input)
        return self.output

    def backprop(self, X, y, alpha):
        error = mse_derivative(self.output, y)
        output_error = sigmoid_derivative(self.output) * error
        weights_error = np.dot(X.T, output_error)
        self.weights -= weights_error * alpha
        self.biases -= output_error.mean() * alpha


def train(input_model, alpha, X, y):
    out = input_model.forward(X)
    input_model.backprop(X, y, alpha)
    mse_res = mse(out, y)
    return mse_res


def get_accuracy(input_model, X_test, y_test):
    out = input_model.forward(X_test)
    max_vals = np.argmax(out, axis=1)
    answers = one_hot(max_vals, vals=TARGETS)
    true_ans = np.sum(np.all(answers == y_test, axis=1))
    total_ans = y_test.shape[0]
    return true_ans / total_ans


def mse(predicted, target):
    return np.mean((predicted - target) ** 2)


def mse_derivative(predicted, target):
    return 2 * (predicted - target) / len(predicted)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xavier(n_in, n_out):
    limit = (np.sqrt(6.0) / np.sqrt(n_in + n_out))
    # Use this return for local run
    # return np.random.uniform(-limit, limit, size=(n_in, n_out))
    return np.random.uniform(-limit, limit, shape=(n_in, n_out))


def scale(X_train, X_test):
    X_train_new = X_train / np.amax(X_train)
    X_test_new = X_test / np.amax(X_test)
    return X_train_new, X_test_new


def one_hot(data: np.ndarray, vals: int = None) -> np.ndarray:
    if not vals:
        vals = data.max() + 1
    y_train = np.zeros((data.size, vals))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):
    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    # Stage 1
    X_train, X_test = scale(X_train, X_test)

    # model = OneLayerNeural(X_train.shape[1], TARGETS)
    model_2 = TwoLayerNeural(X_train.shape[1], TARGETS)

    accuracies = []
    losses = []
    for i in range(20):
        start, end = 0, BATCH_SIZE
        steps = 0
        loss = 0
        while start < len(X_train):
            loss += train(model_2, ALPHA, X_train[start:end], y_train[start:end])
            start += BATCH_SIZE
            end += BATCH_SIZE
            steps += 1
        losses.append(loss / steps)
        accuracy = get_accuracy(model_2, X_test, y_test)
        accuracies.append(accuracy)

    plot(losses, accuracies)

    print(accuracies)
