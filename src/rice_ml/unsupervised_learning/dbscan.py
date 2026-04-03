"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

This module provides a from-scratch implementation of DBSCAN using NumPy.
DBSCAN forms clusters by growing dense regions and marks low-density
points as noise.

Key properties
--------------
- No need to specify the number of clusters in advance
- Can discover non-spherical cluster shapes
- Explicitly labels outliers as noise
- Relies on neighborhood radius and density thresholds

This implementation is written for clarity and instructional value,
without relying on scikit-learn.
"""

import numpy as np


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _validate_feature_matrix(X):
    """
    Validate and coerce the input feature matrix.

    Parameters
    ----------
    X : array_like of shape (n_samples, n_features)
        Input data to cluster.

    Returns
    -------
    ndarray
        Two-dimensional floating-point feature matrix.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    return X


# ---------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------

class DBSCAN:
    """
    Density-based clustering with explicit noise detection.

    Parameters
    ----------
    eps : float, default=0.5
        Neighborhood radius used to decide whether two points are close.
    min_samples : int, default=5
        Minimum number of points required for a point to be considered
        a core point.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels learned during fitting. Noise points receive `-1`.
    n_clusters_ : int
        Number of discovered clusters, excluding noise.
    """

    def __init__(self, eps=0.5, min_samples=5):
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("min_samples must be a positive integer.")

        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _region_query(self, X, sample_index):
        """
        Return the indices of all points within eps of one sample.
        """
        deltas = X - X[sample_index]
        squared_distances = np.sum(deltas * deltas, axis=1)
        return np.flatnonzero(squared_distances <= self.eps ** 2)

    def _expand_cluster(self, X, labels, visited, sample_index, neighbors, cluster_id):
        """
        Grow a cluster outward from an initial core point.
        """
        labels[sample_index] = cluster_id
        queue = list(neighbors)
        head = 0

        while head < len(queue):
            neighbor_index = queue[head]
            head += 1

            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                neighbor_neighbors = self._region_query(X, neighbor_index)
                if neighbor_neighbors.size >= self.min_samples:
                    for candidate in neighbor_neighbors:
                        if candidate not in queue:
                            queue.append(int(candidate))

            if labels[neighbor_index] == -1:
                labels[neighbor_index] = cluster_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Fit DBSCAN clustering to the input data.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
            Fitted estimator.
        """
        X = _validate_feature_matrix(X)
        n_samples = X.shape[0]

        visited = np.zeros(n_samples, dtype=bool)
        labels = np.full(n_samples, -1, dtype=int)
        cluster_id = 0

        for sample_index in range(n_samples):
            if visited[sample_index]:
                continue

            visited[sample_index] = True
            neighbors = self._region_query(X, sample_index)

            if neighbors.size < self.min_samples:
                labels[sample_index] = -1
                continue

            self._expand_cluster(X, labels, visited, sample_index, neighbors, cluster_id)
            cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        return self

    def fit_predict(self, X):
        """
        Fit the model and return cluster labels.
        """
        self.fit(X)
        return self.labels_.copy()
