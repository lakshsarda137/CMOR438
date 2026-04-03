"""
Logistic Regression (Binary Classification)

This module implements a binary logistic regression classifier trained
with batch gradient descent. It includes probability prediction,
decision scores, accuracy, and ROC-curve utilities.
"""

import numpy as np

from ._validation import check_X_y, ensure_2d_numeric, ensure_1d_array


def _sigmoid(z):
    """
    Numerically stable sigmoid function.
    """
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    """
    Binary logistic regression classifier.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Gradient descent step size.
    max_iter : int, default=1000
        Maximum number of optimization iterations.
    tol : float, default=1e-6
        Tolerance for early stopping based on parameter change.
    fit_intercept : bool, default=True
        Whether to estimate an intercept term.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Learned feature weights.
    intercept_ : float
        Learned intercept term.
    classes_ : ndarray of shape (2,)
        Original class labels seen during fitting.
    loss_history_ : list of float
        Binary cross-entropy values over optimization steps.
    """

    def __init__(self, learning_rate=0.1, max_iter=1000, tol=1e-6, fit_intercept=True):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.loss_history_ = []
        self.n_features_in_ = None

    def __repr__(self):
        return (
            "LogisticRegression("
            f"learning_rate={self.learning_rate}, "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol}, "
            f"fit_intercept={self.fit_intercept}"
            ")"
        )

    def _encode_y(self, y):
        """
        Encode arbitrary binary labels as {0, 1}.
        """
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("y must contain exactly two classes for binary logistic regression.")
        encoded = (y == classes[1]).astype(float)
        return encoded, classes

    def _linear_response(self, X):
        """
        Compute the linear predictor.
        """
        return X @ self.coef_ + self.intercept_

    def fit(self, X, y):
        """
        Fit binary logistic regression with batch gradient descent.
        """
        X_arr, y_arr = check_X_y(X, y, allow_1d_X=True)
        y_encoded, self.classes_ = self._encode_y(y_arr)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.max_iter):
            scores = self._linear_response(X_arr)
            probs_pos = _sigmoid(scores)
            errors = probs_pos - y_encoded

            grad_w = X_arr.T @ errors / n_samples
            grad_b = float(errors.mean()) if self.fit_intercept else 0.0

            new_coef = self.coef_ - self.learning_rate * grad_w
            new_intercept = self.intercept_ - self.learning_rate * grad_b

            probs_clipped = np.clip(probs_pos, 1e-12, 1.0 - 1e-12)
            loss = -np.mean(
                y_encoded * np.log(probs_clipped) + (1.0 - y_encoded) * np.log(1.0 - probs_clipped)
            )
            self.loss_history_.append(float(loss))

            step_size = np.linalg.norm(new_coef - self.coef_) + abs(new_intercept - self.intercept_)
            self.coef_ = new_coef
            if self.fit_intercept:
                self.intercept_ = new_intercept

            if step_size <= self.tol:
                break

        return self

    def decision_function(self, X):
        """
        Compute signed decision scores before the sigmoid transform.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit before decision_function.")

        X_arr = ensure_2d_numeric(X, name="X", allow_1d=True)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data.")
        return self._linear_response(X_arr)

    def predict_proba(self, X):
        """
        Return class probabilities for each input sample.
        """
        scores = self.decision_function(X)
        probs_pos = _sigmoid(scores)
        probs_neg = 1.0 - probs_pos
        return np.column_stack((probs_neg, probs_pos))

    def predict(self, X):
        """
        Predict class labels for new samples.
        """
        probs = self.predict_proba(X)[:, 1]
        labels = np.where(probs >= 0.5, self.classes_[1], self.classes_[0])
        return labels

    def score(self, X, y):
        """
        Compute classification accuracy.
        """
        y_true = ensure_1d_array(y, name="y")
        y_pred = self.predict(X)
        return float(np.mean(y_true == y_pred))

    def roc_curve(self, X, y):
        """
        Compute ROC curve coordinates and area under the curve.

        Returns
        -------
        tuple
            `(fpr, tpr, auc)` where `fpr` and `tpr` are NumPy arrays.
        """
        y_true = ensure_1d_array(y, name="y")
        if self.classes_ is None:
            raise RuntimeError("Call fit before roc_curve.")

        y_encoded = (y_true == self.classes_[1]).astype(int)
        scores = self.predict_proba(X)[:, 1]

        thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
        tprs = []
        fprs = []

        positives = np.sum(y_encoded == 1)
        negatives = np.sum(y_encoded == 0)

        for threshold in thresholds:
            preds = (scores >= threshold).astype(int)
            tp = np.sum((preds == 1) & (y_encoded == 1))
            fp = np.sum((preds == 1) & (y_encoded == 0))
            tpr = tp / positives if positives > 0 else 0.0
            fpr = fp / negatives if negatives > 0 else 0.0
            tprs.append(tpr)
            fprs.append(fpr)

        fprs = np.asarray(fprs, dtype=float)
        tprs = np.asarray(tprs, dtype=float)
        order = np.argsort(fprs)
        fprs = fprs[order]
        tprs = tprs[order]
        auc = float(np.trapezoid(tprs, fprs))
        return fprs, tprs, auc
