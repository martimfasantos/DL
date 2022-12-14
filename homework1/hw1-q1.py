#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Sign function
        y_hat = np.argmax(self.W.dot(x_i))
        print(self.W.shape)
        print("x")
        print(x_i.shape)
        print("=")
        print(self.W.dot(x_i).shape)
        print("-----")
        if y_hat != y_i:
            # Update weights
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Label scores (num_labels x 1)
        label_scores = self.W.dot(x_i)[:, None]
        # One-hot vector with the true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1
        # Sigmoid function (num_labels x 1)
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # SGD update (num_labels x num_features=)
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None,:]


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        W1 = np.random.normal(0.1, 0.1, size=(hidden_size, n_features))
        b1 = np.zeros(hidden_size)
        W2 = np.random.normal(0.1, 0.1, size=(n_classes, hidden_size))
        b2 = np.zeros(n_classes)
        print(W1.shape)
        print(W2.shape)
        self.weights = [W1, W2]
        self.biases = [b1, b2]
        self.num_layers = len(self.weights) # layers + 1

    def relu(self, x):
        return np.maximum(0, x)


    def forward(self, x):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        g = self.relu # activation function for hidden layer
        hiddens = []
        for i in range(self.num_layers):
            h = x if i == 0 else hiddens[i-1]
            z = self.weights[i].dot(h) + self.biases[i]
            if i < self.num_layers-1:
                hiddens.append(g(z))
        output = z # last z is the output
        return output, hiddens

    def compute_label_probabilities(self, output):
        # Compute softmax transformation.
        # TODO overflow aqui ?
        probs = np.exp(output) / np.sum(np.exp(output))
        return probs

    def compute_loss(self, output, y):
        probs = self.compute_label_probabilities(output)
        loss = -y.dot(np.log(probs))
        return loss  
    
    def backward(self, x, y, output, hiddens):
        z = output
        # print(z)

        # Activation function for hidden layer
        g = self.relu

        # Grad of loss wrt last z (output).
        probs = self.compute_label_probabilities(output)
        grad_z = probs - y

        grad_weights = []
        grad_biases = []
        for i in range(self.num_layers-1, -1, -1):

            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]
            grad_weights.append(grad_z[:, None].dot(h[:, None].T)) # dL/dW
            grad_biases.append(grad_z) # dL/db

            # Gradient of hidden layer below.
            grad_h = self.weights[i].T.dot(grad_z) # dL/dh

            # Gradient of hidden layer below before activation.
            assert(g == self.relu)
            grad_z = grad_h * (1-h**2) # Grad of loss wrt z

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def update_parameters(self, grad_weights, grad_biases, eta):
        for i in range(self.num_layers):
            self.weights[i] -= eta*grad_weights[i]
            self.biases[i] -= eta*grad_biases[i]

    def predict_label(output):
        y_hat = np.zeros_like(output)
        y_hat[np.argmax(output)] = 1
        return y_hat

    def predict(self, X):
        predicted_labels = []
        for x in X:
            output, _ = self.forward(x)
            y_hat = self.predict_label(output)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat, _ = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            output, hiddens = self.forward(x_i)
            # print(loss = self.compute_loss(output, y))
            grad_weights, grad_biases = self.backward(x_i, y_i, output, hiddens)
            self.update_parameters(grad_weights, grad_biases, learning_rate)


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
