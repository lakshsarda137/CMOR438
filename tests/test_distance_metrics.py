"""
Unit tests for distance_metrics.py.

These tests cover correctness on simple vectors and validation behavior
for malformed inputs.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.distance_metrics import euclidean_distance, manhattan_distance


class TestDistanceMetrics:
    """Tests for Euclidean and Manhattan distance."""

    def test_euclidean_distance_zero_for_identical_vectors(self):
        assert euclidean_distance([1, 2, 3], [1, 2, 3]) == 0.0

    def test_euclidean_distance_basic_case(self):
        assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_manhattan_distance_basic_case(self):
        assert manhattan_distance([1, 2], [4, 6]) == pytest.approx(7.0)

    def test_distance_requires_matching_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            euclidean_distance([1, 2], [1, 2, 3])

    def test_distance_requires_1d_vectors(self):
        with pytest.raises(ValueError, match="1D"):
            manhattan_distance(np.array([[1, 2]]), np.array([[1, 2]]))

    def test_distance_requires_non_empty_vectors(self):
        with pytest.raises(ValueError, match="at least one value"):
            euclidean_distance([], [])
