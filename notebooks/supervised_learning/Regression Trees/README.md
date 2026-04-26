# Regression Trees

This directory contains a detailed Brainvita notebook for **regression trees** using your custom `rice_ml` implementation. It predicts `num_legal_moves` as a continuous mobility score and compares the tree against a mean predictor.

## Notebook

- `regression_trees_brainvita.ipynb`

## Models Used

- `rice_ml.supervised_learning.RegressionTree`

## Dataset

- `data/brainvita_dataset.csv`
- This notebook predicts `num_legal_moves` directly.

## Coverage

1. Variance-reduction split criterion
2. Brainvita regression target and feature rationale
3. Mean-predictor baseline comparison
4. Train/test regression metrics (R2/MSE/RMSE/MAE)
5. Predicted-vs-actual and residual diagnostics
6. Bias-variance interpretation for tree depth and leaf size

## Notes

- Notebook is designed to run end-to-end from the notebook directory or repository root.
- Local path logic resolves `src/rice_ml` robustly from notebook location.
