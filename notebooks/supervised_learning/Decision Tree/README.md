# Decision Tree

This directory contains a detailed Brainvita notebook for a CART-style **decision tree classifier** using your custom `rice_ml` implementation. It classifies high-mobility board states and compares the tree against a majority-class baseline.

## Notebook

- `decision_tree_brainvita.ipynb`

## Models Used

- `rice_ml.supervised_learning.DecisionTreeClassifier`

## Dataset

- `data/brainvita_dataset.csv`
- Classification notebooks use `num_legal_moves >= 3` as the positive class.
- Regression notebooks predict `num_legal_moves` directly.

## Coverage

1. Gini impurity and split-selection intuition
2. Brainvita target framing and feature rationale
3. Majority-class baseline comparison
4. Train/test metrics and confusion matrix
5. Probability distribution diagnostics
6. Overfitting and leaf-probability interpretation

## Notes

- Notebook is designed to run end-to-end from the notebook directory or repository root.
- Local path logic resolves `src/rice_ml` robustly from notebook location.
