"""
Distance metrics for nearest-neighbor style methods.

These helpers are intentionally simple and explicit so they can serve
as both building blocks and teaching examples.
"""

import numpy as np


def _to_1d_float_array(x, name):
    """
    Validate a single observation vector.

    Parameters
    ----------
    x : array_like
        Input vector.
    name : str
        Name used in error messages.

    Returns
    -------
    ndarray
        One-dimensional float array.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return arr


def _validate_pair(a, b):
    """
    Validate a pair of vectors for pointwise distance computation.
    """
    a_arr = _to_1d_float_array(a, "a")
    b_arr = _to_1d_float_array(b, "b")

    if a_arr.shape != b_arr.shape:
        raise ValueError("a and b must have the same shape.")

    return a_arr, b_arr


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors.
    """
    a_arr, b_arr = _validate_pair(a, b)
    return float(np.linalg.norm(a_arr - b_arr))


def manhattan_distance(a, b):
    """
    Compute Manhattan distance between two vectors.
    """
    a_arr, b_arr = _validate_pair(a, b)
    return float(np.sum(np.abs(a_arr - b_arr)))
