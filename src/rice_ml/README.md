# rice_ml

`rice_ml` is a small from-scratch machine learning package built for coursework, experimentation, and algorithmic clarity. The implementations are NumPy-based and intentionally readable so the math behind each model is visible in the code.

This package is not meant to replace production libraries such as scikit-learn. It is meant to make core machine learning ideas inspectable and testable.

## Package Structure

```text
rice_ml/
├── supervised_learning/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── gradient_descent.py
│   ├── knn.py
│   ├── distance_metrics.py
│   ├── perceptron.py
│   ├── multilayer_perceptron.py
│   ├── decision_tree.py
│   ├── regression_trees.py
│   ├── ensemble_methods.py
│   └── _validation.py
│
└── unsupervised_learning/
    ├── k_means.py
    ├── dbscan.py
    ├── pca.py
    └── community_detection.py
```

## Design Goals

- Implement classic algorithms from first principles.
- Keep APIs close to common `fit`, `predict`, `transform`, and `score` patterns.
- Use explicit input validation so shape and type errors fail early.
- Prefer transparent logic over clever abstractions.
- Keep modules independent enough to be tested in isolation.

## Supervised Learning

`supervised_learning` contains models that learn from labeled data.

Included algorithms and utilities:

- `LinearRegression`: ordinary least squares and related regression fitting
- `LogisticRegression`: binary classification with sigmoid probabilities
- `GradientDescent1D` and `GradientDescentND`: optimization helpers
- `KNNClassifier` and `KNNRegressor`: distance-based prediction
- `Perceptron`: mistake-driven linear classification
- `MultilayerPerceptron`: feedforward neural network for binary classification
- `DecisionTreeClassifier`: CART-style classification tree
- `RegressionTree`: CART-style regression tree
- `BaggingClassifier`, `VotingClassifier`, `RandomForestClassifier`, `RandomForestRegressor`: ensemble models
- `euclidean_distance` and `manhattan_distance`: reusable distance functions

## Unsupervised Learning

`unsupervised_learning` contains algorithms that discover structure without target labels.

Included algorithms:

- `KMeans`: centroid-based clustering
- `DBSCAN`: density-based clustering with noise detection
- `PCA`: variance-preserving dimensionality reduction
- `LabelPropagation`: graph-based community detection

## Current Package Boundaries

This package currently keeps preprocessing and evaluation helpers either inside notebooks/tests or inside model modules where needed. Some sample repositories include a separate `processing/` package for shared `train_test_split`, scaling, and metric helpers. That layer is not currently part of this package.

If a shared preprocessing/evaluation API is added later, it should live in a dedicated `rice_ml.processing` subpackage rather than being duplicated inside individual notebooks.

## Testing and Examples

- Unit tests live in `tests/`.
- Supervised notebook examples live in `notebooks/supervised_learning/`.
- Unsupervised notebook examples live in `notebooks/unsupervised_learning/` when available.
