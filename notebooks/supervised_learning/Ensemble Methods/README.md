# Ensemble Methods

This directory contains a detailed Brainvita notebook for **ensemble methods** using your custom `rice_ml` implementation. It compares simple baselines, a single tree, bagging, random forest, and voting for mobility classification, then uses a random forest regressor for the raw legal-move count.

## Notebook

- `ensemble_methods_brainvita.ipynb`

## Models Used

- `rice_ml.supervised_learning.BaggingClassifier`
- `rice_ml.supervised_learning.VotingClassifier`
- `rice_ml.supervised_learning.RandomForestClassifier`
- `rice_ml.supervised_learning.RandomForestRegressor`

## Dataset

- `data/brainvita_dataset.csv`
- Classification notebooks use `num_legal_moves >= 3` as the positive class.
- Regression notebooks predict `num_legal_moves` directly.

## Coverage

1. Variance-reduction motivation for ensembles
2. Brainvita dataset and target framing
3. Majority-class and single-tree baselines
4. Classification ensemble comparison
5. ROC/AUC diagnostics for probability models
6. Regression ensemble evaluation against a mean baseline
7. Majority vote, bootstrap, and random-feature ideas

## Notes

- Notebook is designed to be executable end-to-end.
- Local path logic resolves `src/rice_ml` robustly from notebook location.
