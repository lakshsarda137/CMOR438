"""
Multilayer Perceptron (MLP) for binary classification.

This module implements a fully connected neural network with one or more
hidden layers, ReLU activations, a sigmoid output layer, and batch
gradient descent training.
"""

import numpy as np

from ._validation import check_X_y, ensure_2d_numeric


def sigmoid(z):
    """
    Numerically stable sigmoid activation.
    """
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    """
    ReLU activation.
    """
    return np.maximum(0.0, z)


def relu_derivative(a):
    """
    ReLU derivative expressed in terms of activations.
    """
    return (a > 0.0).astype(float)


class MultilayerPerceptron:
    """
    Multilayer Perceptron for binary classification.

    Parameters
    ----------
    hidden_layers : sequence of int
        Sizes of the hidden layers.
    learning_rate : float, default=0.01
        Batch gradient descent step size.
    max_iter : int, default=1000
        Maximum number of optimization iterations.
    tol : float, default=1e-6
        Early-stopping tolerance based on loss change.
    random_state : int or None, default=None
        Random seed for reproducible initialization.
    """

    def __init__(self, hidden_layers, learning_rate=0.01, max_iter=1000, tol=1e-6, random_state=None):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.hidden_layers = list(hidden_layers)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.weights_ = None
        self.biases_ = None
        self.loss_history_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self._rng = np.random.default_rng(random_state)

    def _encode_y(self, y):
        """
        Encode arbitrary binary labels as {0, 1}.
        """
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("y must contain exactly two classes for the MLP.")
        encoded = (y == classes[1]).astype(float).reshape(-1, 1)
        return encoded, classes

    def _initialize_parameters(self, n_features):
        """
        Initialize weights with scaled random values.
        """
        layer_sizes = [n_features] + self.hidden_layers + [1]
        self.weights_ = []
        self.biases_ = []

        for input_dim, output_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = 1.0 / np.sqrt(max(1, input_dim))
            weights = self._rng.normal(loc=0.0, scale=scale, size=(input_dim, output_dim))
            biases = np.zeros((1, output_dim), dtype=float)
            self.weights_.append(weights)
            self.biases_.append(biases)

    def _forward(self, X):
        """
        Run a forward pass through the network.

        Returns
        -------
        list
            Activations for all layers, including the input.
        """
        activations = [X]

        for layer_idx, (weights, biases) in enumerate(zip(self.weights_, self.biases_)):
            z = activations[-1] @ weights + biases
            if layer_idx == len(self.weights_) - 1:
                activations.append(sigmoid(z))
            else:
                activations.append(relu(z))

        return activations

    def fit(self, X, y):
        """
        Fit the network with batch gradient descent.
        """
        X_arr, y_arr = check_X_y(X, y, allow_1d_X=True)
        y_encoded, self.classes_ = self._encode_y(y_arr)

        self.n_features_in_ = X_arr.shape[1]
        self._initialize_parameters(self.n_features_in_)
        self.loss_history_ = []

        previous_loss = None

        for _ in range(self.max_iter):
            activations = self._forward(X_arr)
            output = activations[-1]
            output_clipped = np.clip(output, 1e-12, 1.0 - 1e-12)
            loss = -np.mean(
                y_encoded * np.log(output_clipped) + (1.0 - y_encoded) * np.log(1.0 - output_clipped)
            )
            self.loss_history_.append(float(loss))

            if previous_loss is not None and abs(previous_loss - loss) <= self.tol:
                break
            previous_loss = float(loss)

            delta = output - y_encoded
            grad_weights = [None] * len(self.weights_)
            grad_biases = [None] * len(self.biases_)

            for layer_idx in range(len(self.weights_) - 1, -1, -1):
                grad_weights[layer_idx] = activations[layer_idx].T @ delta / X_arr.shape[0]
                grad_biases[layer_idx] = np.mean(delta, axis=0, keepdims=True)

                if layer_idx > 0:
                    delta = (delta @ self.weights_[layer_idx].T) * relu_derivative(activations[layer_idx])

            for layer_idx in range(len(self.weights_)):
                self.weights_[layer_idx] -= self.learning_rate * grad_weights[layer_idx]
                self.biases_[layer_idx] -= self.learning_rate * grad_biases[layer_idx]

        return self

    def predict_proba(self, X):
        """
        Return class probabilities for each sample.
        """
        if self.weights_ is None or self.biases_ is None:
            raise RuntimeError("Call fit before predict_proba.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")

        probs_pos = self._forward(X_arr)[-1].ravel()
        probs_neg = 1.0 - probs_pos
        return np.column_stack((probs_neg, probs_pos))

    def predict(self, X):
        """
        Predict binary class labels.
        """
        probs = self.predict_proba(X)[:, 1]
        return np.where(probs >= 0.5, self.classes_[1], self.classes_[0])
