# Logistic Regression

This directory contains a full-depth Brainvita logistic-regression notebook using your custom `rice_ml` implementation. It classifies board states as high mobility when `num_legal_moves >= 3`.

## Notebook

- `logistic_regression_brainvita.ipynb`

## Model

- `rice_ml.supervised_learning.LogisticRegression`

## Coverage

1. Log-odds/sigmoid/log-loss derivation
2. Brainvita classification target and class-balance context
3. Majority-class baseline
4. Hyperparameter sweep and model selection
5. Train/test metrics and confusion matrices
6. Loss-curve diagnostics
7. ROC/AUC and threshold sensitivity
8. Coefficient and odds-ratio interpretation

## Notes

- Positive class is `num_legal_moves >= 3`.
- Designed to run end-to-end from the notebook directory or repository root.
