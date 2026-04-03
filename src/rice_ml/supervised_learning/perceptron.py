"""
Perceptron classifier.

This is the classic single-layer perceptron for binary classification.
It is a foundational linear classifier and a useful stepping stone
toward more sophisticated neural network models.
"""

import numpy as np

from ._validation import check_X_y, ensure_2d_numeric


class Perceptron:
    """
    Binary perceptron classifier.

    Parameters
    ----------
    learning_rate : float, default=1.0
        Size of the update applied on each mistake.
    max_iter : int, default=1000
        Maximum number of full passes through the dataset.
    random_state : int or None, default=None
        Included for API symmetry; training is deterministic here.
    """

    def __init__(self, learning_rate=1.0, max_iter=1000, random_state=None):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.errors_ = []
        self.n_iter_ = 0
        self.n_features_in_ = None

    def _encode_y(self, y):
        """
        Map arbitrary binary labels to {-1, +1}.
        """
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("y must contain exactly two classes for the perceptron.")
        encoded = np.where(y == classes[0], -1.0, 1.0)
        return encoded, classes

    def fit(self, X, y):
        """
        Fit the perceptron using the online update rule.
        """
        X_arr, y_arr = check_X_y(X, y, allow_1d_X=True)
        y_encoded, self.classes_ = self._encode_y(y_arr)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0
        self.errors_ = []

        for iteration in range(1, self.max_iter + 1):
            errors = 0
            for x_i, y_i in zip(X_arr, y_encoded):
                margin = y_i * (np.dot(x_i, self.coef_) + self.intercept_)
                if margin <= 0.0:
                    self.coef_ += self.learning_rate * y_i * x_i
                    self.intercept_ += self.learning_rate * y_i
                    errors += 1
            self.errors_.append(errors)
            self.n_iter_ = iteration
            if errors == 0:
                break

        return self

    def decision_function(self, X):
        """
        Compute signed decision scores.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit before decision_function.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")
        return X_arr @ self.coef_ + self.intercept_

    def predict(self, X):
        """
        Predict class labels.
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
