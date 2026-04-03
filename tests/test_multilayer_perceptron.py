"""
Unit tests for multilayer_perceptron.py.

These tests emphasize stable binary-classification behavior on small
datasets, along with probability semantics and validation.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.multilayer_perceptron import MultilayerPerceptron


class TestMultilayerPerceptron:
    """Tests for the MultilayerPerceptron model."""

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
        model = MultilayerPerceptron(
            hidden_layers=[4],
            learning_rate=0.1,
            max_iter=5000,
            tol=1e-10,
            random_state=0,
        ).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_predict_proba_shape_and_sum(self, or_data):
        X, y = or_data
        model = MultilayerPerceptron(
            hidden_layers=[4],
            learning_rate=0.1,
            max_iter=3000,
            random_state=0,
        ).fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (4, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_loss_history_is_recorded(self, or_data):
        X, y = or_data
        model = MultilayerPerceptron(hidden_layers=[3], learning_rate=0.1, max_iter=50, random_state=0).fit(X, y)
        assert len(model.loss_history_) >= 1

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict_proba"):
            MultilayerPerceptron(hidden_layers=[3]).predict([[0.0, 0.0]])

    def test_non_binary_targets_raise(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="exactly two classes"):
            MultilayerPerceptron(hidden_layers=[3]).fit(X, y)

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MultilayerPerceptron(hidden_layers=[3], learning_rate=0.0)

    def test_invalid_max_iter_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MultilayerPerceptron(hidden_layers=[3], max_iter=0)

    def test_feature_mismatch_raises(self, or_data):
        X, y = or_data
        model = MultilayerPerceptron(hidden_layers=[3], random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="different number of features"):
            model.predict(np.array([[1.0, 2.0, 3.0]]))
