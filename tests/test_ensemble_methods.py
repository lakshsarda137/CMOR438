"""
Unit tests for ensemble_methods.py.

These tests cover bootstrap aggregation, hard voting, and random forests
for both classification and regression.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.decision_tree import DecisionTreeClassifier
from rice_ml.supervised_learning.ensemble_methods import (
    BaggingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from rice_ml.supervised_learning.knn import KNNClassifier
from rice_ml.supervised_learning.logistic_regression import LogisticRegression


class TestBaggingClassifier:
    """Tests for BaggingClassifier."""

    @pytest.fixture
    def class_data(self):
        X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
            [6.0, 6.0],
        ])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        return X, y

    def test_fit_and_predict(self, class_data):
        X, y = class_data
        model = BaggingClassifier(
            base_learner=lambda: DecisionTreeClassifier(max_depth=2, random_state=0),
            n_estimators=7,
            random_state=0,
        ).fit(X, y)
        preds = model.predict(X)
        assert np.mean(preds == y) >= 0.75

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            BaggingClassifier().predict([[0.0, 0.0]])

    def test_invalid_n_estimators_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BaggingClassifier(n_estimators=0)

    def test_invalid_max_samples_raises(self):
        with pytest.raises(ValueError, match="in \\(0, 1\\]"):
            BaggingClassifier(max_samples=0.0)


class TestVotingClassifier:
    """Tests for VotingClassifier."""

    @pytest.fixture
    def class_data(self):
        X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        y = np.array([0, 0, 1, 1])
        return X, y

    def test_fit_and_predict(self, class_data):
        X, y = class_data
        models = [
            LogisticRegression(learning_rate=0.5, max_iter=3000),
            KNNClassifier(n_neighbors=1),
        ]
        voter = VotingClassifier(models).fit(X, y)
        preds = voter.predict(X)
        assert preds.shape == y.shape

    def test_empty_models_raise(self):
        with pytest.raises(ValueError, match="at least one estimator"):
            VotingClassifier([])

    def test_predict_before_fit_raises(self, class_data):
        X, y = class_data
        models = [KNNClassifier(n_neighbors=1)]
        voter = VotingClassifier(models)
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            voter.predict(X)


class TestRandomForestClassifier:
    """Tests for RandomForestClassifier."""

    @pytest.fixture
    def class_data(self):
        rng = np.random.default_rng(0)
        X_left = rng.normal(loc=0.0, scale=0.2, size=(20, 2))
        X_right = rng.normal(loc=3.0, scale=0.2, size=(20, 2))
        X = np.vstack([X_left, X_right])
        y = np.array([0] * 20 + [1] * 20)
        return X, y

    def test_fit_and_predict(self, class_data):
        X, y = class_data
        forest = RandomForestClassifier(n_estimators=11, max_depth=4, random_state=0).fit(X, y)
        preds = forest.predict(X)
        assert np.mean(preds == y) >= 0.9

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            RandomForestClassifier().predict([[0.0, 0.0]])

    def test_invalid_n_estimators_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RandomForestClassifier(n_estimators=0)


class TestRandomForestRegressor:
    """Tests for RandomForestRegressor."""

    @pytest.fixture
    def reg_data(self):
        X = np.arange(0.0, 10.0).reshape(-1, 1)
        y = 2.0 * X.ravel() + 1.0
        return X, y

    def test_fit_and_predict(self, reg_data):
        X, y = reg_data
        forest = RandomForestRegressor(n_estimators=15, max_depth=5, random_state=0).fit(X, y)
        preds = forest.predict(X)
        mse = np.mean((preds - y) ** 2)
        assert mse < 5.0

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before predict"):
            RandomForestRegressor().predict([[0.0]])

    def test_invalid_n_estimators_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RandomForestRegressor(n_estimators=0)
