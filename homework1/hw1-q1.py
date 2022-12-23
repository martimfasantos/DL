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

# Q1.1a
class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Sign function
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            # Update weights
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i

# Q1.1b
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
        # Softmax function (num_labels x 1)
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # SGD update (num_labels x num_features)
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None,:]

# Q1.2b
class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP.
        self.weights = []
        self.biases = []
        
        for i in range(layers + 1):
            if i == 0: # input to l1
                self.weights.append(np.random.normal(0.1, 0.1, size=(hidden_size, n_features)))
                self.biases.append(np.zeros(hidden_size))
            elif i == layers: # last layer to output
                self.weights.append(np.random.normal(0.1, 0.1, size=(n_classes, hidden_size)))
                self.biases.append(np.zeros(n_classes))
            else:
                self.weights.append(np.random.normal(0.1, 0.1, size=(hidden_size, hidden_size)))
                self.biases.append(np.zeros(hidden_size))

        self.num_layers = len(self.weights) # layers + 1

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return 1. * (x > 0)

    def forward_propagation(self, x):
        # Activation function for hidden layers
        g = self.relu

        hiddens = []
        for i in range(self.num_layers):
            h = x if i == 0 else hiddens[i-1]
            z = self.weights[i].dot(h) + self.biases[i]
            if i < self.num_layers-1:
                hiddens.append(g(z))
        output = z # last z is the output
        return output, hiddens

    def softmax(self, x):
        # Compute label probabilities.
        probs = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
        return probs

    def compute_y_one_hot(self, n_classes, y):
        y_one_hot = np.zeros(n_classes)
        y_one_hot[y] = 1
        return y_one_hot

    def compute_loss(self, output, y):
        probs = self.softmax(output)
        loss = -y.dot(np.log(probs))
        return loss
    
    def back_propagation(self, x, y, output, hiddens):
        # Activation function for hidden layer
        g = self.relu

        # Grad of loss wrt last z (output).
        probs = self.softmax(output)
        grad_z = probs - y

        grad_weights = []
        grad_biases = []
        for i in range(self.num_layers-1, -1, -1):

            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]
            grad_weights.append(grad_z[:, None].dot(h[None, :])) # dL/dW
            grad_biases.append(grad_z) # dL/db

            # Gradient of hidden layer below.
            grad_h = self.weights[i].T.dot(grad_z) # dL/dh

            # Gradient of hidden layer below before activation.
            assert(g == self.relu)
            grad_z = grad_h * self.relu_grad(h) # Grad of loss wrt z

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def update_parameters(self, grad_weights, grad_biases, eta):
        for i in range(self.num_layers):
            self.weights[i] -= eta * grad_weights[i]
            self.biases[i] -= eta * grad_biases[i]

    def predict_label(self, output):
        y_hat = np.zeros_like(output)
        y_hat[np.argmax(output)] = 1
        return y_hat

    def predict(self, X):
        predicted_labels = []
        for x in X:
            output, _ = self.forward_propagation(x)
            y_hat = self.predict_label(output)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        predicted_labels = self.predict(X)
        accuracy = np.mean(np.argmax(predicted_labels, axis=1) == y)
        return accuracy

    def train_epoch(self, X, y, learning_rate=0.001):
        total_loss = 0
        for x_i, y_i in zip(X, y):            
            output, hiddens = self.forward_propagation(x_i)
            y_i = self.compute_y_one_hot(len(output), y_i)
            loss = self.compute_loss(output, y_i)
            total_loss += loss
            grad_weights, grad_biases = self.back_propagation(x_i, y_i, output, hiddens)
            self.update_parameters(grad_weights, grad_biases, learning_rate)
        # print("Total loss: %f" % total_loss)
        

def save_plot(model):
    if model == 'perceptron':
        plt.savefig('results/Q1/Q1.1a.pdf', bbox_inches='tight')
    elif model == 'logistic_regression':
        plt.savefig('results/Q1/Q1.1b.pdf', bbox_inches='tight')
    else:
        plt.savefig('results/Q1/Q1.2b.pdf', bbox_inches='tight')
   
        
def plot(model, epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    save_plot(model)
    # plt.show()


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

    # plot & save
    plot(opt.model, epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
