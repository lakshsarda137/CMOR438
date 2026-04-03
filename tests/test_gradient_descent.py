"""
Unit tests for gradient_descent.py.

These tests check convergence on simple quadratic objectives and verify
basic validation behavior.
"""

import numpy as np
import pytest

from rice_ml.supervised_learning.gradient_descent import GradientDescent1D, GradientDescentND


class TestGradientDescent1D:
    """Tests for scalar gradient descent."""

    def test_converges_on_quadratic(self):
        gd = GradientDescent1D(learning_rate=0.1, max_iter=2000, tol=1e-10)
        optimum = gd.optimize(lambda x: 2 * (x - 3), initial_x=0.0)
        assert optimum == pytest.approx(3.0, abs=1e-4)
        assert gd.x_ == pytest.approx(3.0, abs=1e-4)

    def test_history_is_recorded(self):
        gd = GradientDescent1D(learning_rate=0.1, max_iter=10)
        gd.optimize(lambda x: 2 * x, initial_x=1.0)
        assert len(gd.history_) >= 1

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            GradientDescent1D(learning_rate=0.0)


class TestGradientDescentND:
    """Tests for vector gradient descent."""

    def test_converges_on_quadratic_bowl(self):
        gd = GradientDescentND(learning_rate=0.1, max_iter=3000, tol=1e-10)
        optimum = gd.optimize(lambda x: 2 * (x - np.array([1.0, -2.0])), initial_x=np.array([0.0, 0.0]))
        assert np.allclose(optimum, np.array([1.0, -2.0]), atol=1e-4)

    def test_requires_1d_initial_point(self):
        gd = GradientDescentND()
        with pytest.raises(ValueError, match="1D array"):
            gd.optimize(lambda x: x, initial_x=np.array([[1.0, 2.0]]))

    def test_gradient_output_shape_must_match_initial_point(self):
        gd = GradientDescentND()
        with pytest.raises(ValueError, match="must match"):
            gd.optimize(lambda x: np.array([1.0]), initial_x=np.array([1.0, 2.0]))
