"""
Validation helpers shared across supervised learning modules.

These utilities keep error messages and coercion rules consistent across
the package. They are intentionally lightweight and depend only on NumPy.
"""

import numpy as np


def ensure_2d_numeric(X, name="X", allow_1d=False):
    """
    Validate a numeric feature matrix and return a float NumPy array.

    Parameters
    ----------
    X : array_like
        Candidate feature matrix.
    name : str, default="X"
        Name used in error messages.
    allow_1d : bool, default=False
        If True, a one-dimensional input is interpreted as a single
        feature and reshaped to `(n_samples, 1)`.

    Returns
    -------
    ndarray
        Two-dimensional floating-point array.
    """
    X = np.asarray(X)

    if allow_1d and X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    try:
        return X.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc


def ensure_1d_array(y, name="y"):
    """
    Validate a one-dimensional target array.

    Parameters
    ----------
    y : array_like
        Candidate target vector.
    name : str, default="y"
        Name used in error messages.

    Returns
    -------
    ndarray
        One-dimensional NumPy array.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if y.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return y


def ensure_1d_numeric(y, name="y"):
    """
    Validate a one-dimensional numeric target vector.
    """
    y = ensure_1d_array(y, name=name)
    try:
        return y.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc


def check_X_y(X, y, *, y_numeric=False, allow_1d_X=False):
    """
    Validate a supervised learning design matrix and target vector.

    Parameters
    ----------
    X : array_like
        Feature matrix.
    y : array_like
        Target vector.
    y_numeric : bool, default=False
        If True, coerce `y` to float.
    allow_1d_X : bool, default=False
        If True, reshape 1D `X` to `(n_samples, 1)`.

    Returns
    -------
    tuple
        `(X_arr, y_arr)` after validation and coercion.
    """
    X_arr = ensure_2d_numeric(X, name="X", allow_1d=allow_1d_X)
    y_arr = ensure_1d_numeric(y, name="y") if y_numeric else ensure_1d_array(y, name="y")

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    return X_arr, y_arr
