# Multilayer Perceptron

This directory contains a detailed Brainvita notebook for a **multilayer perceptron** using your custom `rice_ml` implementation. It compares the MLP against a perceptron baseline and diagnoses both optimization and probability separation.

## Notebook

- `multilayer_perceptron_brainvita.ipynb`

## Models Used

- `rice_ml.supervised_learning.MultilayerPerceptron`
- `rice_ml.supervised_learning.Perceptron`

## Dataset

- `data/brainvita_dataset.csv`
- Classification notebooks use `num_legal_moves >= 3` as the positive class.
- Regression notebooks predict `num_legal_moves` directly.

## Coverage

1. Hidden-layer, ReLU, sigmoid, and backprop intuition
2. Brainvita high-mobility target framing
3. Train-only standardization for gradient descent
4. Perceptron baseline comparison
5. MLP train/test metrics and confusion matrices
6. Loss-curve and probability-separation diagnostics

## Notes

- Notebook is designed to run end-to-end from the notebook directory or repository root.
- Local path logic resolves `src/rice_ml` robustly from notebook location.
