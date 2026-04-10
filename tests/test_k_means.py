"""
Unit tests for k_means.py.

These tests validate API behavior, error handling, and basic clustering
correctness on simple synthetic datasets.
"""

import numpy as np
import pytest

from rice_ml.unsupervised_learning.k_means import KMeans


class TestKMeans:
    """Unit tests for KMeans."""

    @pytest.fixture
    def two_cluster_data(self):
        return np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [8.0, 8.0],
            [8.0, 9.0],
            [9.0, 8.0],
        ])

    def test_fit_finds_two_clusters(self, two_cluster_data):
        model = KMeans(n_clusters=2, random_state=0)
        labels = model.fit_predict(two_cluster_data)

        assert labels.shape == (6,)
        assert len(np.unique(labels)) == 2
        assert model.cluster_centers_.shape == (2, 2)
        assert model.inertia_ >= 0.0

    def test_fit_sets_attributes(self, two_cluster_data):
        model = KMeans(n_clusters=2, random_state=0).fit(two_cluster_data)

        assert model.cluster_centers_ is not None
        assert model.labels_ is not None
        assert model.inertia_ is not None
        assert model.n_iter_ >= 1

    def test_predict_after_fit(self, two_cluster_data):
        model = KMeans(n_clusters=2, random_state=0).fit(two_cluster_data)
        predictions = model.predict(np.array([[0.5, 0.5], [8.5, 8.5]]))

        assert predictions.shape == (2,)
        assert predictions[0] != predictions[1]

    def test_predict_matches_training_labels(self, two_cluster_data):
        model = KMeans(n_clusters=2, random_state=0).fit(two_cluster_data)
        predictions = model.predict(two_cluster_data)

        np.testing.assert_array_equal(predictions, model.labels_)

    def test_transform_returns_distances_to_centers(self, two_cluster_data):
        model = KMeans(n_clusters=2, random_state=0).fit(two_cluster_data)
        distances = model.transform(two_cluster_data[:2])
        assert distances.shape == (2, 2)
        assert np.all(distances >= 0.0)

    def test_score_is_negative_inertia(self, two_cluster_data):
        model = KMeans(n_clusters=2, random_state=0).fit(two_cluster_data)
        assert model.score(two_cluster_data) == pytest.approx(-model.inertia_)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            KMeans(n_clusters=2).predict(np.array([[0.0, 0.0]]))

    def test_repr(self):
        model = KMeans(n_clusters=3, max_iter=50, tol=1e-3, random_state=7)
        assert "KMeans(" in repr(model)
        assert "n_clusters=3" in repr(model)

    def test_too_many_clusters_raises(self):
        X = np.array([[0.0], [1.0]])
        with pytest.raises(ValueError, match="at least n_clusters"):
            KMeans(n_clusters=3).fit(X)

    def test_invalid_n_clusters_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            KMeans(n_clusters=0)

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError, match="2D array"):
            KMeans(n_clusters=2).fit(np.array([1.0, 2.0, 3.0]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="at least one sample"):
            KMeans(n_clusters=2).fit(np.empty((0, 2)))

    def test_separable_clusters_have_distant_centers(self):
        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.normal(loc=0.0, scale=0.15, size=(20, 2)),
            rng.normal(loc=5.0, scale=0.15, size=(20, 2)),
        ])

        model = KMeans(n_clusters=2, random_state=0).fit(X)
        center_distance = np.linalg.norm(
            model.cluster_centers_[0] - model.cluster_centers_[1]
        )

        assert len(np.unique(model.labels_)) == 2
        assert center_distance > 3.0
