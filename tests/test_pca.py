"""
Unit tests for pca.py.

These tests validate PCA shape behavior, numerical properties,
reconstruction behavior, and error handling.
"""

import numpy as np
import pytest

from rice_ml.unsupervised_learning.pca import PCA


class TestPCA:
    """Unit tests for PCA."""

    @pytest.fixture
    def correlated_data(self):
        return np.array([
            [1.0, 2.0],
            [2.0, 4.1],
            [3.0, 6.0],
            [4.0, 8.2],
            [5.0, 10.1],
        ])

    @pytest.fixture
    def high_dim_data(self):
        rng = np.random.default_rng(1)
        return rng.normal(size=(50, 5))

    def test_fit_stores_components_and_variance(self, correlated_data):
        model = PCA(n_components=1).fit(correlated_data)

        assert model.components_.shape == (1, 2)
        assert model.explained_variance_.shape == (1,)
        assert model.explained_variance_ratio_.shape == (1,)
        assert np.isclose(model.explained_variance_ratio_.sum(), model.explained_variance_ratio_[0])

    def test_fit_returns_self(self, correlated_data):
        model = PCA(n_components=1)
        out = model.fit(correlated_data)
        assert out is model

    def test_fit_transform_reduces_dimension(self, correlated_data):
        transformed = PCA(n_components=1).fit_transform(correlated_data)
        assert transformed.shape == (5, 1)

    def test_transform_after_fit_has_expected_shape(self, correlated_data):
        model = PCA(n_components=2).fit(correlated_data)
        transformed = model.transform(correlated_data)

        assert transformed.shape == (5, 2)

    def test_inverse_transform_restores_shape(self, correlated_data):
        model = PCA(n_components=1).fit(correlated_data)
        transformed = model.transform(correlated_data)
        reconstructed = model.inverse_transform(transformed)

        assert reconstructed.shape == correlated_data.shape

    def test_repr(self):
        assert repr(PCA(n_components=3)) == "PCA(n_components=3)"

    def test_explained_variance_is_sorted(self, correlated_data):
        model = PCA(n_components=2).fit(correlated_data)
        ev = model.explained_variance_

        assert ev[0] >= ev[1]

    def test_explained_variance_ratio_sums_to_one_when_all_components_kept(self, high_dim_data):
        model = PCA(n_components=5).fit(high_dim_data)
        assert pytest.approx(model.explained_variance_ratio_.sum(), rel=1e-6) == 1.0

    def test_components_are_orthonormal(self, high_dim_data):
        model = PCA(n_components=5).fit(high_dim_data)
        identity = model.components_ @ model.components_.T
        assert np.allclose(identity, np.eye(5), atol=1e-6)

    def test_full_reconstruction_is_nearly_exact(self, high_dim_data):
        model = PCA(n_components=5).fit(high_dim_data)
        transformed = model.transform(high_dim_data)
        reconstructed = model.inverse_transform(transformed)
        mse = np.mean((high_dim_data - reconstructed) ** 2)

        assert mse < 1e-10

    def test_float_n_components_selects_components(self, correlated_data):
        model = PCA(n_components=0.9).fit(correlated_data)
        assert model.n_components_ in (1, 2)
        assert model.n_components_ >= 1

    def test_transform_before_fit_raises(self, correlated_data):
        with pytest.raises(RuntimeError, match="Call fit before transform"):
            PCA(n_components=1).transform(correlated_data)

    def test_invalid_n_components_raises(self, correlated_data):
        with pytest.raises(ValueError, match="between 1 and n_features"):
            PCA(n_components=3).fit(correlated_data)

    def test_zero_n_components_raises(self, correlated_data):
        with pytest.raises(ValueError, match="between 1 and n_features"):
            PCA(n_components=0).fit(correlated_data)

    def test_invalid_float_n_components_raises(self, correlated_data):
        with pytest.raises(ValueError, match="float n_components"):
            PCA(n_components=1.5).fit(correlated_data)

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError, match="2D array"):
            PCA(n_components=1).fit(np.array([1.0, 2.0, 3.0]))

    def test_inverse_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="Call fit before inverse_transform"):
            PCA(n_components=1).inverse_transform(np.array([[1.0], [2.0]]))
