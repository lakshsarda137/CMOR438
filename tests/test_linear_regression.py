"""
Unit tests for linear_regression.py.

These tests cover fitting, prediction, metric computation, and common
validation scenarios.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.linear_regression import LinearRegression


class TestLinearRegression:
    """Tests for the LinearRegression model."""

    @pytest.fixture
    def simple_data(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y = 2.0 * X.ravel() + 1.0
        return X, y

    def test_fit_recovers_simple_line(self, simple_data):
        X, y = simple_data
        model = LinearRegression().fit(X, y)

        assert model.coef_[0] == pytest.approx(2.0, abs=1e-8)
        assert model.intercept_ == pytest.approx(1.0, abs=1e-8)

    def test_predict_returns_expected_shape(self, simple_data):
        X, y = simple_data
        model = LinearRegression().fit(X, y)
        preds = model.predict(X)

        assert preds.shape == y.shape

    def test_fit_returns_self(self, simple_data):
        X, y = simple_data
        model = LinearRegression()
        assert model.fit(X, y) is model

    def test_r2_score_is_one_on_perfect_line(self, simple_data):
        X, y = simple_data
        model = LinearRegression().fit(X, y)
        assert model.r2_score(X, y) == pytest.approx(1.0)

    def test_mse_and_rmse_are_zero_on_perfect_fit(self, simple_data):
        X, y = simple_data
        model = LinearRegression().fit(X, y)
        assert model.mse(X, y) == pytest.approx(0.0)
        assert model.rmse(X, y) == pytest.approx(0.0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            LinearRegression().predict([[0.0]])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of samples"):
            LinearRegression().fit([[0.0], [1.0]], [1.0])

    def test_non_numeric_X_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            LinearRegression().fit([["a"], ["b"]], [1.0, 2.0])

    def test_fit_without_intercept(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model = LinearRegression(fit_intercept=False).fit(X, y)

        assert model.intercept_ == pytest.approx(0.0)
        assert model.coef_[0] == pytest.approx(2.0)
