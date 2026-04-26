# Source Code

This directory contains the core Python source code for the repository. It is split into two packages:

```text
src/
├── brainvita/
└── rice_ml/
```

## `brainvita`

`brainvita` contains the game-specific code used to generate and work with Brainvita board states.

Main areas:

- `board_constructor/`: board representation and visualization helpers
- `solver/`: move logic and solver utilities
- `data_generator/`: feature extraction and dataset generation
- `reinforcement_learning/`: environment and agent scaffolding

## `rice_ml`

`rice_ml` contains the from-scratch machine learning implementations used by the notebooks and tests.

Main areas:

- `supervised_learning/`: regression, classification, trees, nearest neighbors, neural-network, and ensemble models
- `unsupervised_learning/`: clustering, dimensionality reduction, and graph-based community detection

The package is educational by design: implementations favor clarity, explicit validation, and notebook-friendly behavior over production-scale optimization.

## Development Notes

- Tests live in `tests/` and cover both `brainvita` and `rice_ml`.
- Notebooks in `notebooks/` demonstrate how the package modules are used in a narrative format.
- Source modules should avoid notebook-specific assumptions so they remain reusable from scripts, tests, and examples.
