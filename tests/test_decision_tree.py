"""
Unit tests for decision_tree.py.

These tests validate simple classification behavior, splitting logic,
and input validation.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.decision_tree import DecisionTree, DecisionTreeClassifier


class TestDecisionTreeClassifier:
    """Tests for DecisionTreeClassifier."""

    @pytest.fixture
    def threshold_data(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        return X, y

    def test_fit_learns_simple_threshold(self, threshold_data):
        X, y = threshold_data
        model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_alias_decision_tree_works(self, threshold_data):
        X, y = threshold_data
        model = DecisionTree(max_depth=2, random_state=0).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            DecisionTreeClassifier().predict([[0.0]])

    def test_max_depth_limits_tree_but_still_predicts(self, threshold_data):
        X, y = threshold_data
        model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_invalid_max_depth_raises(self):
        with pytest.raises(ValueError, match="positive or None"):
            DecisionTreeClassifier(max_depth=0)

    def test_invalid_min_samples_split_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            DecisionTreeClassifier(min_samples_split=1)

    def test_invalid_min_samples_leaf_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            DecisionTreeClassifier(min_samples_leaf=0)

    def test_feature_mismatch_raises(self, threshold_data):
        X, y = threshold_data
        model = DecisionTreeClassifier(random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="different number of features"):
            model.predict(np.array([[1.0, 2.0]]))

    def test_constant_features_produce_leaf(self):
        X = np.ones((4, 2))
        y = np.array([0, 0, 1, 1])
        model = DecisionTreeClassifier(random_state=0).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_float_max_features_requires_valid_range(self, threshold_data):
        X, y = threshold_data
        with pytest.raises(ValueError, match="max_features"):
            DecisionTreeClassifier(max_features=1.5).fit(X, y)
