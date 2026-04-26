# k-Nearest Neighbors (KNN)

This directory contains a detailed Brainvita notebook for **k-nearest neighbors (KNN)** using your custom `rice_ml` implementation. It uses KNN as a similarity-based model for both high-mobility classification and legal-move regression.

## Notebook

- `knn_brainvita.ipynb`

## Models Used

- `rice_ml.supervised_learning.KNNClassifier`
- `rice_ml.supervised_learning.KNNRegressor`

## Dataset

- `data/brainvita_dataset.csv`
- Classification notebooks use `num_legal_moves >= 3` as the positive class.
- Regression notebooks predict `num_legal_moves` directly.

## Coverage

1. Distance-based KNN intuition
2. Brainvita classification and regression targets
3. Train-only standardization for distance metrics
4. Majority-class and mean-predictor baselines
5. k-sensitivity analysis with accuracy and F1
6. Regression metrics and residual diagnostics

## Notes

- Notebook is designed to run end-to-end from the notebook directory or repository root.
- Local path logic resolves `src/rice_ml` robustly from notebook location.
