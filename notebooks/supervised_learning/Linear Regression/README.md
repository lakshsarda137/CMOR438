# Linear Regression

This directory contains a full-depth notebook for OLS linear regression on Brainvita using your custom `rice_ml` implementation. The notebook treats `num_legal_moves` as a board-state mobility score and builds an interpretable baseline before moving to nonlinear methods.

## Notebook

- `linear_regression_brainvita.ipynb`

## Model

- `rice_ml.supervised_learning.LinearRegression`

## What This Notebook Covers

1. OLS derivation and assumptions
2. Brainvita dataset story and feature-selection rationale
3. Leakage-safe split and standardization
4. Baseline comparisons before full model
5. Full model training and metric table
6. Residual diagnostics and assumption checks
7. Coefficient interpretation and sensitivity analysis

## Notes

- Predicts `num_legal_moves` as a regression target.
- Designed to run end-to-end from the notebook directory or repository root.
