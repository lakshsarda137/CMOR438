"""
K-Means Clustering (From Scratch)

This module implements the K-Means clustering algorithm using NumPy.
K-Means is an unsupervised learning method that partitions data into
K clusters by minimizing the within-cluster sum of squares.

Features
--------
- Random centroid initialization from observed samples
- Euclidean-distance cluster assignment
- Mean-based centroid updates
- Convergence detection via centroid movement
- Inertia computation for fit quality inspection

This implementation is intended to be readable, educational, and
consistent with the rest of the rice_ml package.
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
# K-Means clustering
# ---------------------------------------------------------------------

class KMeans:
    """
    K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of Lloyd iterations.
    tol : float, default=1e-4
        Convergence tolerance based on centroid movement.
    random_state : int or None, default=None
        Random seed used for centroid initialization.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Learned cluster centroids.
    labels_ : ndarray of shape (n_samples,)
        Cluster assignment of each training sample.
    inertia_ : float
        Sum of squared distances from each sample to its assigned centroid.
    n_iter_ : int
        Number of iterations performed during fitting.
    """

    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            "KMeans("
            f"n_clusters={self.n_clusters}, "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol}, "
            f"random_state={self.random_state}"
            ")"
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_centers(self, X):
        """
        Initialize centroids by sampling distinct observations.
        """
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be at least n_clusters.")
        indices = self._rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_distances(X, centers):
        """
        Compute Euclidean distances from each sample to each center.
        """
        deltas = X[:, None, :] - centers[None, :, :]
        return np.linalg.norm(deltas, axis=2)

    def _assign_clusters(self, X, centers):
        """
        Assign each sample to the nearest centroid and compute inertia.
        """
        distances = self._pairwise_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        squared_distances = distances[np.arange(X.shape[0]), labels] ** 2
        inertia = float(np.sum(squared_distances))
        return labels, inertia

    def _update_centers(self, X, labels):
        """
        Recompute centroids as the mean of their assigned samples.
        """
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=float)

        for cluster_id in range(self.n_clusters):
            members = X[labels == cluster_id]
            if members.shape[0] == 0:
                # Re-seed empty clusters with a random observed point.
                centers[cluster_id] = X[self._rng.integers(0, X.shape[0])]
            else:
                centers[cluster_id] = members.mean(axis=0)

        return centers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Fit K-Means clustering to the input data.

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
        centers = self._initialize_centers(X)

        for iteration in range(1, self.max_iter + 1):
            labels, inertia = self._assign_clusters(X, centers)
            updated_centers = self._update_centers(X, labels)
            shift = np.linalg.norm(updated_centers - centers)
            centers = updated_centers

            self.n_iter_ = iteration
            if shift <= self.tol:
                break

        labels, inertia = self._assign_clusters(X, centers)
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def predict(self, X):
        """
        Assign new samples to the nearest learned centroid.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Data to label.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted cluster labels.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit before predict.")
        X = _validate_feature_matrix(X)
        labels, _ = self._assign_clusters(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X):
        """
        Fit the model and return training-set cluster labels.
        """
        self.fit(X)
        return self.labels_.copy()
