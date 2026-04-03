"""
Basic gradient descent utilities.

These classes are not predictors themselves; they are small numerical
optimization helpers that mirror the educational utilities seen in
several of the sample repositories.
"""

import numpy as np


class GradientDescent1D:
    """
    Gradient descent for a scalar-valued one-dimensional objective.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size used for each update.
    max_iter : int, default=1000
        Maximum number of gradient steps.
    tol : float, default=1e-8
        Convergence tolerance based on gradient magnitude.
    """

    def __init__(self, learning_rate=0.1, max_iter=1000, tol=1e-8):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.history_ = []
        self.x_ = None
        self.n_iter_ = 0

    def optimize(self, gradient, initial_x):
        """
        Minimize an objective using its derivative.

        Parameters
        ----------
        gradient : callable
            Derivative of the objective with respect to x.
        initial_x : float
            Starting point.

        Returns
        -------
        float
            Final iterate after optimization.
        """
        x = float(initial_x)
        self.history_ = [x]

        for iteration in range(1, self.max_iter + 1):
            grad = float(gradient(x))
            if abs(grad) <= self.tol:
                self.n_iter_ = iteration - 1
                self.x_ = x
                return x
            x -= self.learning_rate * grad
            self.history_.append(x)

        self.n_iter_ = self.max_iter
        self.x_ = x
        return x


class GradientDescentND:
    """
    Gradient descent for a multi-parameter objective.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Step size used for each update.
    max_iter : int, default=1000
        Maximum number of gradient steps.
    tol : float, default=1e-8
        Convergence tolerance based on gradient norm.
    """

    def __init__(self, learning_rate=0.1, max_iter=1000, tol=1e-8):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.history_ = []
        self.x_ = None
        self.n_iter_ = 0

    def optimize(self, gradient, initial_x):
        """
        Minimize an objective using its vector-valued gradient.

        Parameters
        ----------
        gradient : callable
            Function returning the gradient at the current iterate.
        initial_x : array_like
            Starting point.

        Returns
        -------
        ndarray
            Final iterate after optimization.
        """
        x = np.asarray(initial_x, dtype=float)
        if x.ndim != 1:
            raise ValueError("initial_x must be a 1D array.")

        self.history_ = [x.copy()]

        for iteration in range(1, self.max_iter + 1):
            grad = np.asarray(gradient(x), dtype=float)
            if grad.shape != x.shape:
                raise ValueError("gradient output must match the shape of initial_x.")
            if np.linalg.norm(grad) <= self.tol:
                self.n_iter_ = iteration - 1
                self.x_ = x.copy()
                return x
            x = x - self.learning_rate * grad
            self.history_.append(x.copy())

        self.n_iter_ = self.max_iter
        self.x_ = x.copy()
        return x
