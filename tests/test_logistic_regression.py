"""
Unit tests for logistic_regression.py.

These tests validate binary classification behavior, probability
predictions, ROC-curve computation, and error handling.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.logistic_regression import LogisticRegression


class TestLogisticRegression:
    """Tests for the LogisticRegression model."""

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
        model = LogisticRegression(learning_rate=0.5, max_iter=4000, tol=1e-8).fit(X, y)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_predict_proba_shape_and_row_sums(self, or_data):
        X, y = or_data
        model = LogisticRegression(learning_rate=0.5, max_iter=3000).fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (4, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_decision_function_orders_examples_reasonably(self, or_data):
        X, y = or_data
        model = LogisticRegression(learning_rate=0.5, max_iter=3000).fit(X, y)
        scores = model.decision_function(X)

        assert scores[0] < scores[-1]

    def test_score_returns_accuracy(self, or_data):
        X, y = or_data
        model = LogisticRegression(learning_rate=0.5, max_iter=3000).fit(X, y)
        assert model.score(X, y) == pytest.approx(1.0)

    def test_loss_history_is_recorded(self, or_data):
        X, y = or_data
        model = LogisticRegression(learning_rate=0.5, max_iter=100).fit(X, y)
        assert len(model.loss_history_) >= 1

    def test_roc_curve_outputs_valid_values(self, or_data):
        X, y = or_data
        model = LogisticRegression(learning_rate=0.5, max_iter=3000).fit(X, y)
        fpr, tpr, auc = model.roc_curve(X, y)

        assert fpr.ndim == 1
        assert tpr.ndim == 1
        assert fpr.shape == tpr.shape
        assert np.all((0.0 <= fpr) & (fpr <= 1.0))
        assert np.all((0.0 <= tpr) & (tpr <= 1.0))
        assert 0.0 <= auc <= 1.0

    def test_non_binary_targets_raise(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="exactly two classes"):
            LogisticRegression().fit(X, y)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before decision_function"):
            LogisticRegression().predict([[0.0]])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of samples"):
            LogisticRegression().fit([[0.0], [1.0]], [0])

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            LogisticRegression(learning_rate=0.0)
