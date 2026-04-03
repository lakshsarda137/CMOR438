"""
Ensemble learning methods.

This module collects a small set of ensemble models built from the
package's own supervised learners:

- BaggingClassifier
- VotingClassifier
- RandomForestClassifier
- RandomForestRegressor
"""

import numpy as np

from ._validation import check_X_y, ensure_2d_numeric
from .decision_tree import DecisionTreeClassifier
from .regression_trees import RegressionTree


def _majority_vote(labels):
    """
    Return the most common label in a one-dimensional collection.
    """
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


class BaggingClassifier:
    """
    Bootstrap aggregating classifier.

    Parameters
    ----------
    base_learner : callable, default=DecisionTreeClassifier
        Callable that returns a fresh classifier instance.
    n_estimators : int, default=10
        Number of bootstrap models to fit.
    max_samples : float, default=1.0
        Fraction of the training set sampled for each estimator.
    random_state : int or None, default=None
        Random seed for bootstrap sampling.
    """

    def __init__(self, base_learner=DecisionTreeClassifier, n_estimators=10, max_samples=1.0, random_state=None):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if not 0.0 < max_samples <= 1.0:
            raise ValueError("max_samples must be in (0, 1].")

        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.models_ = []
        self.n_features_in_ = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """
        Fit an ensemble of base learners on bootstrap samples.
        """
        X_arr, y_arr = check_X_y(X, y, allow_1d_X=True)
        self.n_features_in_ = X_arr.shape[1]
        sample_size = max(1, int(np.ceil(self.max_samples * X_arr.shape[0])))

        self.models_ = []
        for _ in range(self.n_estimators):
            indices = self._rng.choice(X_arr.shape[0], size=sample_size, replace=True)
            model = self.base_learner()
            model.fit(X_arr[indices], y_arr[indices])
            self.models_.append(model)

        return self

    def predict(self, X):
        """
        Predict labels by majority vote across the ensemble.
        """
        if not self.models_:
            raise RuntimeError("Call fit before predict.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")

        predictions = np.asarray([model.predict(X_arr) for model in self.models_], dtype=object)
        return np.asarray([_majority_vote(predictions[:, col]) for col in range(X_arr.shape[0])], dtype=object)


class VotingClassifier:
    """
    Hard-voting ensemble classifier.

    Parameters
    ----------
    models : list
        Instantiated classifiers that expose `fit` and `predict`.
    """

    def __init__(self, models):
        if len(models) == 0:
            raise ValueError("models must contain at least one estimator.")
        self.models = models
        self.n_features_in_ = None

    def fit(self, X, y):
        """
        Fit each constituent model on the same training data.
        """
        X_arr, y_arr = check_X_y(X, y, allow_1d_X=True)
        self.n_features_in_ = X_arr.shape[1]

        for model in self.models:
            model.fit(X_arr, y_arr)

        return self

    def predict(self, X):
        """
        Predict labels by majority vote across the supplied models.
        """
        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if self.n_features_in_ is None:
            raise RuntimeError("Call fit before predict.")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")

        predictions = np.asarray([model.predict(X_arr) for model in self.models], dtype=object)
        return np.asarray([_majority_vote(predictions[:, col]) for col in range(X_arr.shape[0])], dtype=object)


class RandomForestClassifier:
    """
    Random forest classifier built from decision tree classifiers.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum number of samples needed to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples allowed in each leaf.
    max_features : int, float, str, or None, default="sqrt"
        Number of features considered at each split.
    random_state : int or None, default=None
        Random seed for bootstrap sampling and tree feature subsampling.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.trees_ = []
        self.n_features_in_ = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """
        Fit the random forest classifier.
        """
        X_arr, y_arr = check_X_y(X, y, allow_1d_X=True)
        self.n_features_in_ = X_arr.shape[1]
        self.trees_ = []

        for tree_index in range(self.n_estimators):
            indices = self._rng.choice(X_arr.shape[0], size=X_arr.shape[0], replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + tree_index,
            )
            tree.fit(X_arr[indices], y_arr[indices])
            self.trees_.append(tree)

        return self

    def predict(self, X):
        """
        Predict class labels by majority vote across all trees.
        """
        if not self.trees_:
            raise RuntimeError("Call fit before predict.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")

        predictions = np.asarray([tree.predict(X_arr) for tree in self.trees_], dtype=object)
        return np.asarray([_majority_vote(predictions[:, col]) for col in range(X_arr.shape[0])], dtype=object)


class RandomForestRegressor:
    """
    Random forest regressor built from regression trees.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.trees_ = []
        self.n_features_in_ = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """
        Fit the random forest regressor.
        """
        X_arr, y_arr = check_X_y(X, y, y_numeric=True, allow_1d_X=True)
        self.n_features_in_ = X_arr.shape[1]
        self.trees_ = []

        for tree_index in range(self.n_estimators):
            indices = self._rng.choice(X_arr.shape[0], size=X_arr.shape[0], replace=True)
            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + tree_index,
            )
            tree.fit(X_arr[indices], y_arr[indices])
            self.trees_.append(tree)

        return self

    def predict(self, X):
        """
        Predict continuous targets by averaging tree predictions.
        """
        if not self.trees_:
            raise RuntimeError("Call fit before predict.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")

        predictions = np.asarray([tree.predict(X_arr) for tree in self.trees_], dtype=float)
        return np.mean(predictions, axis=0)
