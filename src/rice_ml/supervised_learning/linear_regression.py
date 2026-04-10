"""
Linear Regression (Ordinary Least Squares)

This module implements a compact from-scratch linear regression model
using the normal equation with a pseudo-inverse. The emphasis is on
clarity, predictable behavior, and a scikit-learn-like API.
"""

import numpy as np

from ._validation import check_X_y, ensure_2d_numeric


class LinearRegression:
    """
    Ordinary Least Squares linear regression.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to estimate an intercept term.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated feature weights.
    intercept_ : float
        Estimated intercept term.
    n_features_in_ : int
        Number of input features seen during fitting.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None

    def __repr__(self):
        return f"LinearRegression(fit_intercept={self.fit_intercept})"

    def _augment(self, X):
        """
        Add a bias column when an intercept is being fitted.
        """
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            return np.hstack((ones, X))
        return X

    def fit(self, X, y):
        """
        Fit the linear regression model.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Feature matrix.
        y : array_like of shape (n_samples,)
            Continuous response vector.

        Returns
        -------
        self
            Fitted estimator.
        """
        X_arr, y_arr = check_X_y(X, y, y_numeric=True, allow_1d_X=True)
        X_design = self._augment(X_arr)
        beta = np.linalg.pinv(X_design) @ y_arr

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X):
        """
        Predict targets for new samples.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit before predict.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")
        return X_arr @ self.coef_ + self.intercept_

    def residuals(self, X, y):
        """
        Compute residuals on a dataset.
        """
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        return y_true - y_pred

    def mse(self, X, y):
        """
        Compute mean squared error on a dataset.
        """
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        return float(np.mean((y_true - y_pred) ** 2))

    def rmse(self, X, y):
        """
        Compute root mean squared error on a dataset.
        """
        return float(np.sqrt(self.mse(X, y)))

    def mae(self, X, y):
        """
        Compute mean absolute error on a dataset.
        """
        residuals = self.residuals(X, y)
        return float(np.mean(np.abs(residuals)))

    def r2_score(self, X, y):
        """
        Compute the coefficient of determination.
        """
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        total = np.sum((y_true - y_true.mean()) ** 2)
        residual = np.sum((y_true - y_pred) ** 2)
        if np.isclose(total, 0.0):
            return 1.0 if np.isclose(residual, 0.0) else 0.0
        return float(1.0 - residual / total)

    def score(self, X, y):
        """
        Return the coefficient of determination, R^2.
        """
        return self.r2_score(X, y)
