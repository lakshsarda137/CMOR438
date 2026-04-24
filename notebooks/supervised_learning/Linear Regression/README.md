# Linear Regression

This directory contains a full, math-first walkthrough of **Linear Regression implemented from scratch** with your custom `rice_ml` package and applied to your custom Brainvita dataset.

---

## Notebook Overview

**Notebook:** `linear_regression_brainvita.ipynb`  
**Model:** `rice_ml.supervised_learning.LinearRegression`  
**Dataset:** `data/brainvita_dataset.csv`  
**Prediction Target:** `num_legal_moves`

The notebook is designed to teach both the theory and the implementation details of linear regression in a single narrative.

---

## What This Notebook Covers

1. **Mathematical foundations of OLS**
   - linear model formulation
   - least-squares objective
   - normal equation intuition
   - why pseudo-inverse improves numerical stability

2. **Brainvita problem framing**
   - interpreting board-state engineered features
   - selecting a continuous target (`num_legal_moves`)
   - excluding leakage-prone outputs (`best_move_*`, `worst_move_*`, raw `board_state`)

3. **Data inspection and EDA**
   - shape and schema checks
   - target distribution analysis
   - feature-to-target correlation ranking

4. **Preprocessing pipeline**
   - train/test split
   - train-only standardization to avoid leakage
   - handling near-zero feature variance safely

5. **Model training with custom `rice_ml` implementation**
   - fitting `LinearRegression(fit_intercept=True)`
   - examining intercept and learned coefficients

6. **Evaluation and diagnostics**
   - R², MSE, RMSE, MAE on train/test
   - naive baseline comparison (mean predictor)
   - predicted-vs-actual visualization
   - residual-vs-predicted and residual histogram

7. **Interpretation and limitations**
   - coefficient magnitude interpretation on standardized features
   - linearity limitations for game-state dynamics
   - next-step ideas for nonlinear models

---

## Brainvita Dataset Context

`brainvita_dataset.csv` contains engineered numeric features derived from board states, including:

- peg count and occupancy composition
- mobility and jumpability summaries
- adjacency-based local structure statistics
- cluster and spread descriptors
- board geometry metadata

The notebook treats this as a regression task where we estimate how many legal moves are available from a state.

---

## Why This Notebook Is Structured This Way

The flow mirrors the strongest parts of the sample repositories while staying faithful to your project goals:

- **math first** (not just API usage)
- **implementation clarity** (explicit preprocessing and modeling steps)
- **diagnostics and interpretation** (not only a single metric)
- **reusable template quality** for the remaining supervised techniques

---

## Execution Notes

- The notebook is already executable end-to-end from within this repository.
- Local import logic resolves `src/rice_ml` robustly from notebook location.
- Outputs are saved in-place so readers can immediately inspect both code and results.

---

## Next Relationship to the Full Notebook Set

This notebook is the interpretable baseline for the supervised sequence. Later notebooks (logistic regression, trees, ensembles, MLP, etc.) can be compared against it for improvements in predictive performance versus interpretability.
