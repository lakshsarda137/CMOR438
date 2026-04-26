# Perceptron

This directory contains a full-depth Brainvita perceptron notebook using your custom `rice_ml` implementation. It treats `num_legal_moves >= 3` as a high-mobility classification target and uses the perceptron as a transparent linear baseline before the MLP.

## Notebook

- `perceptron_brainvita.ipynb`

## Model

- `rice_ml.supervised_learning.Perceptron`

## Coverage

1. Perceptron rule and separability intuition
2. Binary target design from `num_legal_moves`
3. Leakage-safe split and train-only scaling
4. Majority-class baseline and hyperparameter sweep
5. Train/test metrics and confusion matrices
6. Convergence, margin, and coefficient diagnostics

## Notes

- Positive class is defined as `num_legal_moves >= 3`.
- Designed to run end-to-end from the notebook directory or repository root.
