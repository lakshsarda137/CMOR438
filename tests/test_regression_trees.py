"""
Unit tests for regression_trees.py.

These tests check that the regression tree can fit simple piecewise
relationships and that it validates common error cases.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.regression_trees import RegressionTree


class TestRegressionTree:
    """Tests for the RegressionTree model."""

    @pytest.fixture
    def piecewise_data(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0])
        return X, y

    def test_fit_learns_piecewise_constant_pattern(self, piecewise_data):
        X, y = piecewise_data
        model = RegressionTree(max_depth=2, random_state=0).fit(X, y)
        preds = model.predict(X)
        assert np.mean((preds - y) ** 2) < 1e-10

    def test_score_is_one_on_piecewise_training_data(self, piecewise_data):
        X, y = piecewise_data
        model = RegressionTree(max_depth=2, random_state=0).fit(X, y)
        assert model.score(X, y) == pytest.approx(1.0)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            RegressionTree().predict([[0.0]])

    def test_invalid_max_depth_raises(self):
        with pytest.raises(ValueError, match="positive or None"):
            RegressionTree(max_depth=0)

    def test_invalid_min_samples_split_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            RegressionTree(min_samples_split=1)

    def test_invalid_min_samples_leaf_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            RegressionTree(min_samples_leaf=0)

    def test_feature_mismatch_raises(self, piecewise_data):
        X, y = piecewise_data
        model = RegressionTree(random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="different number of features"):
            model.predict(np.array([[1.0, 2.0]]))

    def test_non_numeric_target_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            RegressionTree().fit([[0.0], [1.0]], ["a", "b"])

    def test_float_max_features_requires_valid_range(self, piecewise_data):
        X, y = piecewise_data
        with pytest.raises(ValueError, match="max_features"):
            RegressionTree(max_features=1.2).fit(X, y)
