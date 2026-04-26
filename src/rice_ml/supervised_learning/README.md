# Supervised Learning

This package contains from-scratch supervised learning algorithms for labeled datasets. The modules cover regression, binary classification, nearest-neighbor methods, tree models, neural networks, optimization helpers, and ensembles.

## Modules

| Module | Main Exports | Purpose |
| :--- | :--- | :--- |
| `linear_regression.py` | `LinearRegression` | Predict continuous targets with linear models. |
| `logistic_regression.py` | `LogisticRegression` | Predict binary classes with sigmoid probabilities. |
| `gradient_descent.py` | `GradientDescent1D`, `GradientDescentND` | Demonstrate iterative optimization routines. |
| `knn.py` | `KNNClassifier`, `KNNRegressor` | Predict from nearby stored training examples. |
| `distance_metrics.py` | `euclidean_distance`, `manhattan_distance` | Provide distance functions used by KNN-style models. |
| `perceptron.py` | `Perceptron` | Learn a linear classifier through mistake-driven updates. |
| `multilayer_perceptron.py` | `MultilayerPerceptron` | Train a small feedforward neural network for binary classification. |
| `decision_tree.py` | `DecisionTreeClassifier`, `DecisionTree` | Fit CART-style classification trees using impurity reduction. |
| `regression_trees.py` | `RegressionTree` | Fit CART-style regression trees using variance reduction. |
| `ensemble_methods.py` | `BaggingClassifier`, `VotingClassifier`, `RandomForestClassifier`, `RandomForestRegressor` | Combine multiple learners for more stable predictions. |
| `_validation.py` | internal validation helpers | Normalize shape/type checks across models. |

## Public API

The package-level `__init__.py` exposes the main model classes and helper functions:

```python
from rice_ml.supervised_learning import LinearRegression, LogisticRegression
from rice_ml.supervised_learning import KNNClassifier, KNNRegressor
from rice_ml.supervised_learning import DecisionTreeClassifier, RegressionTree
from rice_ml.supervised_learning import RandomForestClassifier, RandomForestRegressor
```

Most estimators follow a familiar pattern:

- `fit(X, y)` trains the model.
- `predict(X)` returns predictions.
- `predict_proba(X)` is available for classifiers that support class probabilities.
- `score(X, y)` is available for several models as a convenience metric.

## Design Notes

- Models use NumPy arrays internally and validate dimensions before training or prediction.
- Implementations prioritize readability and mathematical transparency.
- Tree and nearest-neighbor methods do not require feature scaling, but gradient-based models usually benefit from it.
- Shared preprocessing and metric utilities are not currently centralized in this package; notebooks define small local helpers where needed.

## Notebook Coverage

The supervised notebooks under `notebooks/supervised_learning/` provide narrative examples for the main techniques in this package:

- Linear Regression
- Logistic Regression
- Decision Tree
- Regression Trees
- KNN
- Perceptron
- Multilayer Perceptron
- Ensemble Methods
