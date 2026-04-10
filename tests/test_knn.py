"""
Unit tests for knn.py.

These tests cover both classification and regression variants of the
k-nearest neighbors algorithm.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor


class TestKNNClassifier:
    """Tests for KNNClassifier."""

    @pytest.fixture
    def class_data(self):
        X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
        ])
        y = np.array(["A", "A", "A", "B", "B", "B"])
        return X, y

    def test_fit_returns_self(self, class_data):
        X, y = class_data
        model = KNNClassifier(n_neighbors=1)
        assert model.fit(X, y) is model

    def test_predict_training_labels_with_one_neighbor(self, class_data):
        X, y = class_data
        model = KNNClassifier(n_neighbors=1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_predict_proba_shape(self, class_data):
        X, y = class_data
        model = KNNClassifier(n_neighbors=3).fit(X, y)
        proba = model.predict_proba(X[:2])

        assert proba.shape == (2, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_kneighbors_returns_distances_and_indices(self, class_data):
        X, y = class_data
        model = KNNClassifier(n_neighbors=2).fit(X, y)
        distances, indices = model.kneighbors(X[:1])

        assert distances.shape == (1, 2)
        assert indices.shape == (1, 2)
        assert distances[0, 0] == pytest.approx(0.0)

    def test_score_returns_accuracy(self, class_data):
        X, y = class_data
        model = KNNClassifier(n_neighbors=1).fit(X, y)
        assert model.score(X, y) == pytest.approx(1.0)

    def test_distance_weighting_changes_vote_strength(self, class_data):
        X, y = class_data
        model = KNNClassifier(n_neighbors=3, weights="distance").fit(X, y)
        pred = model.predict(np.array([[0.1, 0.1]]))
        assert pred[0] == "A"

    def test_manhattan_metric_is_supported(self, class_data):
        X, y = class_data
        model = KNNClassifier(n_neighbors=3, metric="manhattan").fit(X, y)
        pred = model.predict(np.array([[5.2, 5.1]]))
        assert pred[0] == "B"

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            KNNClassifier().predict([[0.0, 0.0]])

    def test_invalid_n_neighbors_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            KNNClassifier(n_neighbors=0)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            KNNClassifier(metric="cosine")

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError, match="weights"):
            KNNClassifier(weights="weird")

    def test_feature_mismatch_raises(self, class_data):
        X, y = class_data
        model = KNNClassifier().fit(X, y)
        with pytest.raises(ValueError, match="different number of features"):
            model.predict(np.array([[1.0, 2.0, 3.0]]))

    def test_too_many_neighbors_raises_on_fit(self, class_data):
        X, y = class_data
        with pytest.raises(ValueError, match="cannot exceed"):
            KNNClassifier(n_neighbors=len(y) + 1).fit(X, y)


class TestKNNRegressor:
    """Tests for KNNRegressor."""

    @pytest.fixture
    def reg_data(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        return X, y

    def test_predicts_local_average(self, reg_data):
        X, y = reg_data
        model = KNNRegressor(n_neighbors=2).fit(X, y)
        pred = model.predict(np.array([[1.1]]))
        assert pred[0] == pytest.approx(1.5)

    def test_distance_weighting_for_regression(self, reg_data):
        X, y = reg_data
        model = KNNRegressor(n_neighbors=2, weights="distance").fit(X, y)
        pred = model.predict(np.array([[1.1]]))
        assert 1.0 < pred[0] < 2.0

    def test_regressor_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            KNNRegressor().predict([[0.0]])

    def test_regressor_score_is_perfect_on_training_line(self, reg_data):
        X, y = reg_data
        model = KNNRegressor(n_neighbors=1).fit(X, y)
        assert model.score(X, y) == pytest.approx(1.0)
