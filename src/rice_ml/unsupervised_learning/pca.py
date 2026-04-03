"""
Principal Component Analysis (PCA)

This module provides a from-scratch implementation of Principal
Component Analysis using NumPy only.

PCA is an unsupervised dimensionality reduction technique that projects
data onto orthogonal directions of maximum variance. It is commonly
used for:

- Visualizing high-dimensional data
- Compressing correlated features
- Reducing noise
- Preprocessing data before downstream modeling
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
        Input data for PCA.

    Returns
    -------
    ndarray
        Two-dimensional floating-point feature matrix.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two samples.")
    return X


# ---------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------

class PCA:
    """
    Principal Component Analysis.

    Parameters
    ----------
    n_components : None, int, or float, default=None
        Number of principal components to retain. If `None`, all feature
        directions are kept. If an integer, exactly that many components
        are retained. If a float in `(0, 1]`, enough components are kept
        to explain at least that fraction of total variance.

    Attributes
    ----------
    n_components_ : int
        Actual number of retained components after fitting.
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean used to center the data.
    components_ : ndarray of shape (n_components_, n_features)
        Principal axes in feature space.
    explained_variance_ : ndarray of shape (n_components_,)
        Variance explained by each retained component.
    explained_variance_ratio_ : ndarray of shape (n_components_,)
        Fraction of total variance explained by each retained component.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.n_components_ = None
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"PCA(n_components={self.n_components})"

    # ------------------------------------------------------------------
    # Component selection
    # ------------------------------------------------------------------

    def _resolve_n_components(self, explained_variance_ratio, n_features):
        """
        Determine how many principal components to retain.
        """
        if self.n_components is None:
            return n_features

        if isinstance(self.n_components, int):
            if self.n_components <= 0 or self.n_components > n_features:
                raise ValueError("n_components must be between 1 and n_features.")
            return self.n_components

        if isinstance(self.n_components, float):
            if not 0.0 < self.n_components <= 1.0:
                raise ValueError("float n_components must be in (0, 1].")
            cumulative = np.cumsum(explained_variance_ratio)
            return int(np.searchsorted(cumulative, self.n_components) + 1)

        raise TypeError("n_components must be None, an int, or a float.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Fit PCA to the input data.

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
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_

        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        total_variance = np.sum(eigenvalues)
        if total_variance <= 0:
            raise ValueError("Total variance must be positive.")

        explained_variance_ratio = eigenvalues / total_variance
        n_features = X.shape[1]
        self.n_components_ = self._resolve_n_components(explained_variance_ratio, n_features)
        self.components_ = eigenvectors[:, : self.n_components_].T
        self.explained_variance_ = eigenvalues[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        return self

    def transform(self, X):
        """
        Project data into the learned principal-component space.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Data to project.

        Returns
        -------
        ndarray of shape (n_samples, n_components_)
            Low-dimensional representation of the input data.
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Call fit before transform.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        centered = X - self.mean_
        return centered @ self.components_.T

    def fit_transform(self, X):
        """
        Fit the model and immediately transform the input data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct data from principal-component space.

        Parameters
        ----------
        X_transformed : array_like of shape (n_samples, n_components_)
            Data expressed in principal-component coordinates.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Approximate reconstruction in the original feature space.
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Call fit before inverse_transform.")
        X_transformed = np.asarray(X_transformed, dtype=float)
        if X_transformed.ndim != 2:
            raise ValueError("X_transformed must be a 2D array.")
        return X_transformed @ self.components_ + self.mean_
