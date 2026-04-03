"""
K-Nearest Neighbors for classification and regression.

The implementations here emphasize directness and transparency. Training
stores the dataset, and prediction is based entirely on distances to the
memorized training samples.
"""

import numpy as np

from ._validation import check_X_y, ensure_2d_numeric


class _KNNBase:
    """
    Shared functionality for nearest-neighbor models.
    """

    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform"):
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'.")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

        self.X_train_ = None
        self.y_train_ = None
        self.n_features_in_ = None

    def _distance_matrix(self, X_query):
        """
        Compute pairwise distances from queries to the stored training set.
        """
        if self.metric == "euclidean":
            deltas = X_query[:, None, :] - self.X_train_[None, :, :]
            return np.linalg.norm(deltas, axis=2)

        return np.sum(np.abs(X_query[:, None, :] - self.X_train_[None, :, :]), axis=2)

    def _get_neighbor_info(self, X_query):
        """
        Return neighbor indices and distances for each query sample.
        """
        distances = self._distance_matrix(X_query)
        neighbor_count = min(self.n_neighbors, self.X_train_.shape[0])
        indices = np.argsort(distances, axis=1)[:, :neighbor_count]
        neighbor_distances = np.take_along_axis(distances, indices, axis=1)
        return indices, neighbor_distances

    def _get_weights(self, distances):
        """
        Convert distances into vote or averaging weights.
        """
        if self.weights == "uniform":
            return np.ones_like(distances)
        return 1.0 / (distances + 1e-12)

    def fit(self, X, y):
        """
        Store the training dataset.
        """
        X_arr, y_arr = check_X_y(X, y, allow_1d_X=True)
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self.n_features_in_ = X_arr.shape[1]
        return self

    def _validate_query(self, X):
        """
        Validate query samples before prediction.
        """
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Call fit before predict.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")
        return X_arr


class KNNClassifier(_KNNBase):
    """
    k-nearest neighbors classifier.
    """

    def fit(self, X, y):
        """
        Store the training dataset and class labels.
        """
        super().fit(X, y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict(self, X):
        """
        Predict class labels for each query sample.
        """
        X_arr = self._validate_query(X)
        indices, distances = self._get_neighbor_info(X_arr)
        predictions = []

        for row_idx in range(indices.shape[0]):
            labels = self.y_train_[indices[row_idx]]
            weights = self._get_weights(distances[row_idx])
            class_scores = []
            for cls in self.classes_:
                score = float(np.sum(weights[labels == cls]))
                class_scores.append(score)
            predictions.append(self.classes_[int(np.argmax(class_scores))])

        return np.asarray(predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities based on neighbor votes.
        """
        X_arr = self._validate_query(X)
        indices, distances = self._get_neighbor_info(X_arr)
        probabilities = np.zeros((X_arr.shape[0], self.classes_.shape[0]), dtype=float)

        for row_idx in range(indices.shape[0]):
            labels = self.y_train_[indices[row_idx]]
            weights = self._get_weights(distances[row_idx])
            total_weight = float(np.sum(weights))
            for class_idx, cls in enumerate(self.classes_):
                probabilities[row_idx, class_idx] = np.sum(weights[labels == cls]) / total_weight

        return probabilities


class KNNRegressor(_KNNBase):
    """
    k-nearest neighbors regressor.
    """

    def fit(self, X, y):
        """
        Store numeric training targets for regression.
        """
        X_arr, y_arr = check_X_y(X, y, y_numeric=True, allow_1d_X=True)
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X):
        """
        Predict continuous targets by averaging neighbor responses.
        """
        X_arr = self._validate_query(X)
        indices, distances = self._get_neighbor_info(X_arr)
        predictions = np.zeros(X_arr.shape[0], dtype=float)

        for row_idx in range(indices.shape[0]):
            values = self.y_train_[indices[row_idx]]
            weights = self._get_weights(distances[row_idx])
            predictions[row_idx] = np.sum(weights * values) / np.sum(weights)

        return predictions
