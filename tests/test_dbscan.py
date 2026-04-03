"""
Unit tests for dbscan.py.

These tests check cluster discovery, noise labeling, and basic
input-validation behavior.
"""

import numpy as np
import pytest

from rice_ml.unsupervised_learning.dbscan import DBSCAN


class TestDBSCAN:
    """Unit tests for DBSCAN."""

    @pytest.fixture
    def cluster_data(self):
        return np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [20.0, 20.0],
        ])

    def test_fit_detects_clusters_and_noise(self, cluster_data):
        model = DBSCAN(eps=0.3, min_samples=2)
        labels = model.fit_predict(cluster_data)

        assert labels.shape == (7,)
        assert model.n_clusters_ == 2
        assert labels[-1] == -1

    def test_two_dense_groups_form_two_clusters(self):
        rng = np.random.default_rng(0)
        X_left = rng.normal(loc=0.0, scale=0.1, size=(40, 2))
        X_right = rng.normal(loc=4.5, scale=0.1, size=(40, 2))
        X = np.vstack([X_left, X_right])

        labels = DBSCAN(eps=0.3, min_samples=4).fit_predict(X)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        assert len(unique_labels) == 2

    def test_all_noise_when_density_too_small(self, cluster_data):
        labels = DBSCAN(eps=0.05, min_samples=3).fit_predict(cluster_data)
        np.testing.assert_array_equal(labels, -np.ones(cluster_data.shape[0], dtype=int))

    def test_invalid_eps_raises(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            DBSCAN(eps=0.0)

    def test_invalid_min_samples_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            DBSCAN(min_samples=0)

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError, match="2D array"):
            DBSCAN().fit(np.array([1.0, 2.0, 3.0]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="at least one sample"):
            DBSCAN().fit(np.empty((0, 2)))
