"""
Unit tests for perceptron.py.

These tests check learning behavior on simple binary datasets and verify
basic API semantics.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.perceptron import Perceptron


class TestPerceptron:
    """Tests for the Perceptron model."""

    @pytest.fixture
    def or_data(self):
        X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        y = np.array([0, 1, 1, 1])
        return X, y

    def test_fit_learns_or_function(self, or_data):
        X, y = or_data
        model = Perceptron(max_iter=20).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_decision_function_shape(self, or_data):
        X, y = or_data
        model = Perceptron(max_iter=20).fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (4,)

    def test_errors_history_is_recorded(self, or_data):
        X, y = or_data
        model = Perceptron(max_iter=20).fit(X, y)
        assert len(model.errors_) >= 1
        assert model.n_iter_ >= 1

    def test_score_is_perfect_on_or_data(self, or_data):
        X, y = or_data
        model = Perceptron(max_iter=20, random_state=0).fit(X, y)
        assert model.score(X, y) == pytest.approx(1.0)

    def test_fit_without_intercept_is_supported(self, or_data):
        X, y = or_data
        model = Perceptron(max_iter=20, fit_intercept=False, shuffle=False).fit(X[:, 1:], y)
        preds = model.predict(X[:, 1:])
        assert preds.shape == y.shape

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before decision_function"):
            Perceptron().predict([[0.0, 0.0]])

    def test_non_binary_targets_raise(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="exactly two classes"):
            Perceptron().fit(X, y)

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Perceptron(learning_rate=0.0)

    def test_invalid_max_iter_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Perceptron(max_iter=0)
