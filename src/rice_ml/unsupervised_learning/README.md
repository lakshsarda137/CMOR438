# Unsupervised Learning

This package contains from-scratch unsupervised learning algorithms. These models discover structure without supervised target labels, using similarity, density, variance, or graph connectivity.

## Modules

| Module | Main Export | Purpose |
| :--- | :--- | :--- |
| `k_means.py` | `KMeans` | Partition data into centroid-based clusters. |
| `dbscan.py` | `DBSCAN` | Identify dense regions and mark sparse points as noise. |
| `pca.py` | `PCA` | Reduce dimensionality by projecting onto high-variance directions. |
| `community_detection.py` | `LabelPropagation` | Detect graph communities through iterative label propagation. |

## Public API

The package-level `__init__.py` exposes:

```python
from rice_ml.unsupervised_learning import KMeans, DBSCAN, PCA, LabelPropagation
```

Typical usage follows the estimator pattern:

- `fit(X)` learns structure from data.
- `predict(X)` or labels attributes are available where the algorithm supports assignment.
- `transform(X)` or `fit_transform(X)` are available for dimensionality-reduction workflows such as PCA.

## Design Notes

- Implementations are intended for learning and experimentation rather than production-scale clustering.
- Algorithms use explicit NumPy operations and simple state attributes.
- The modules are independent from supervised models except for shared Python and NumPy conventions.

## Relationship to Notebooks

Notebook examples for unsupervised learning should explain the algorithm intuition, data assumptions, parameter effects, visual diagnostics, and limitations in the same style as the supervised notebook set.
