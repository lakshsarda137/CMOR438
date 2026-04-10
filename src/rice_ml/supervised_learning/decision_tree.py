"""
Decision tree classification.

This module implements a small CART-style decision tree classifier using
Gini impurity and binary threshold splits.
"""

from dataclasses import dataclass
import numpy as np

from ._validation import check_X_y, ensure_2d_numeric


@dataclass
class _TreeNode:
    """
    Single node in a binary decision tree.
    """

    is_leaf: bool
    prediction: object
    feature_index: int = None
    threshold: float = None
    proba: object = None
    left: object = None
    right: object = None


def _gini(y):
    """
    Compute Gini impurity for a label vector.
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1.0 - np.sum(probabilities ** 2)


class DecisionTreeClassifier:
    """
    Binary-split decision tree classifier.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the fitted tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples allowed in each leaf.
    max_features : int, float, str, or None, default=None
        Number of candidate features to consider per split. Supports
        `None`, integer counts, fractions, `"sqrt"`, and `"log2"`.
    random_state : int or None, default=None
        Random seed for feature subsampling.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be positive or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.tree_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self._rng = np.random.default_rng(random_state)

    def _resolve_max_features(self, n_features):
        """
        Resolve the max_features specification to an integer count.
        """
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, int):
            return min(max(1, self.max_features), n_features)
        if isinstance(self.max_features, float):
            if not 0.0 < self.max_features <= 1.0:
                raise ValueError("float max_features must be in (0, 1].")
            return max(1, int(np.ceil(self.max_features * n_features)))
        raise ValueError("Invalid max_features specification.")

    def _majority_class(self, y):
        """
        Return the most common label in y.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _class_proba(self, y):
        """
        Compute class probabilities in the order of self.classes_.
        """
        probabilities = np.zeros(self.classes_.shape[0], dtype=float)
        for class_index, cls in enumerate(self.classes_):
            probabilities[class_index] = np.mean(y == cls)
        return probabilities

    def _best_split(self, X, y):
        """
        Find the feature and threshold that minimize weighted Gini impurity.
        """
        n_samples, n_features = X.shape
        candidate_count = self._resolve_max_features(n_features)
        candidate_features = self._rng.choice(n_features, size=candidate_count, replace=False)

        best_feature = None
        best_threshold = None
        best_score = np.inf

        for feature_index in candidate_features:
            values = np.unique(X[:, feature_index])
            if values.size == 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                n_left = int(np.sum(left_mask))
                n_right = int(np.sum(right_mask))
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_impurity = _gini(y[left_mask])
                right_impurity = _gini(y[right_mask])
                score = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity

                if score < best_score:
                    best_score = score
                    best_feature = feature_index
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        """
        prediction = self._majority_class(y)
        proba = self._class_proba(y)

        if (
            np.unique(y).size == 1
            or X.shape[0] < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return _TreeNode(is_leaf=True, prediction=prediction, proba=proba)

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return _TreeNode(is_leaf=True, prediction=prediction, proba=proba)

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            is_leaf=False,
            prediction=prediction,
            feature_index=feature_index,
            threshold=threshold,
            proba=proba,
            left=left,
            right=right,
        )

    def fit(self, X, y):
        """
        Fit the decision tree classifier.
        """
        X_arr, y_arr = check_X_y(X, y, allow_1d_X=True)
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_arr.shape[1]
        self.tree_ = self._build_tree(X_arr, y_arr, depth=0)
        return self

    def _predict_one(self, x):
        """
        Predict one sample by traversing the fitted tree.
        """
        node = self.tree_
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X):
        """
        Predict class labels for a batch of samples.
        """
        if self.tree_ is None:
            raise RuntimeError("Call fit before predict.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")
        return np.asarray([self._predict_one(row) for row in X_arr])

    def predict_proba(self, X):
        """
        Predict class probabilities by returning leaf class frequencies.
        """
        if self.tree_ is None:
            raise RuntimeError("Call fit before predict_proba.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")

        probabilities = np.asarray([self._traverse_one(row).proba for row in X_arr], dtype=float)
        return probabilities

    def _traverse_one(self, x):
        """
        Traverse the tree and return the terminal node for one sample.
        """
        node = self.tree_
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def score(self, X, y):
        """
        Compute classification accuracy.
        """
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_true == y_pred))


DecisionTree = DecisionTreeClassifier
