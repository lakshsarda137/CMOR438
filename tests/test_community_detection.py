"""
Unit tests for community_detection.py.

These tests validate correctness on small graphs where the community
structure is easy to reason about directly.
"""

import numpy as np
import pytest

from rice_ml.unsupervised_learning.community_detection import LabelPropagation


class TestLabelPropagation:
    """Unit tests for label propagation community detection."""

    @pytest.fixture
    def two_component_graph(self):
        return np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ], dtype=float)

    def test_fit_finds_two_communities(self, two_component_graph):
        model = LabelPropagation(max_iter=50, random_state=0).fit(two_component_graph)

        assert model.labels_.shape == (6,)
        assert len(np.unique(model.labels_[:3])) == 1
        assert len(np.unique(model.labels_[3:])) == 1
        assert model.labels_[0] != model.labels_[3]
        assert model.n_communities_ == 2

    def test_single_dense_component_forms_one_community(self):
        n_nodes = 5
        graph = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)

        model = LabelPropagation(max_iter=50, random_state=0).fit(graph)
        assert len(set(model.labels_)) == 1

    def test_isolated_node_keeps_a_label(self):
        graph = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=float)

        labels = LabelPropagation(random_state=0).fit_predict(graph)
        assert labels.shape == (3,)
        assert labels[0] == labels[1]

    def test_all_isolated_nodes_keep_unique_labels(self):
        graph = np.zeros((3, 3), dtype=float)
        labels = LabelPropagation(max_iter=10, random_state=0).fit_predict(graph)

        assert len(set(labels)) == 3

    def test_invalid_max_iter_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            LabelPropagation(max_iter=0)

    def test_non_square_graph_raises(self):
        graph = np.array([[0, 1, 0], [1, 0, 1]], dtype=float)
        with pytest.raises(ValueError, match="square"):
            LabelPropagation().fit(graph)

    def test_negative_edge_weight_raises(self):
        graph = np.array([[0, -1], [-1, 0]], dtype=float)
        with pytest.raises(ValueError, match="negative values"):
            LabelPropagation().fit(graph)
