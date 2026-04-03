"""
Community Detection Algorithms (From Scratch)

This module implements a simple label propagation method for finding
communities in a graph without relying on graph-learning libraries.

Community detection is useful when the data is naturally represented as
a network rather than as an ordinary feature matrix. In that setting,
the goal is to identify groups of nodes that are more strongly connected
to each other than to the rest of the graph.
"""

import numpy as np


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _validate_adjacency_matrix(adjacency_matrix):
    """
    Validate and coerce an adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : array_like of shape (n_nodes, n_nodes)
        Weighted or unweighted adjacency matrix of an undirected graph.

    Returns
    -------
    ndarray
        Square, non-negative floating-point adjacency matrix.
    """
    adjacency_matrix = np.asarray(adjacency_matrix, dtype=float)
    if adjacency_matrix.ndim != 2:
        raise ValueError("Adjacency matrix must be 2D.")
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if np.any(adjacency_matrix < 0):
        raise ValueError("Adjacency matrix must not contain negative values.")
    return adjacency_matrix


# ---------------------------------------------------------------------
# Label propagation
# ---------------------------------------------------------------------

class LabelPropagation:
    """
    Community detection via the Label Propagation Algorithm.

    Each node begins with a unique label. At each iteration, a node adopts
    the most strongly supported label among its neighbors. Repeating this
    process tends to collapse densely connected regions into a common label.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of label-update passes.
    random_state : int or None, default=None
        Random seed used to shuffle node order and break ties.

    Attributes
    ----------
    labels_ : ndarray of shape (n_nodes,)
        Final community label for each node.
    n_communities_ : int
        Number of distinct labels present at convergence.
    n_iter_ : int
        Number of iterations executed.
    """

    def __init__(self, max_iter=100, random_state=None):
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.n_communities_ = None
        self.n_iter_ = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, adjacency_matrix):
        """
        Fit label propagation to the supplied graph.

        Parameters
        ----------
        adjacency_matrix : array_like of shape (n_nodes, n_nodes)
            Adjacency matrix of the graph.

        Returns
        -------
        self
            Fitted estimator.
        """
        A = _validate_adjacency_matrix(adjacency_matrix)
        n_nodes = A.shape[0]
        rng = np.random.default_rng(self.random_state)

        labels = np.arange(n_nodes, dtype=int)
        node_order = np.arange(n_nodes, dtype=int)

        for iteration in range(1, self.max_iter + 1):
            changed = False
            rng.shuffle(node_order)

            for node_index in node_order:
                neighbor_indices = np.flatnonzero(A[node_index] > 0)
                if neighbor_indices.size == 0:
                    continue

                neighbor_labels = labels[neighbor_indices]
                neighbor_weights = A[node_index, neighbor_indices]

                unique_labels, inverse = np.unique(neighbor_labels, return_inverse=True)
                weighted_votes = np.bincount(inverse, weights=neighbor_weights)
                best_vote = np.max(weighted_votes)
                tied_indices = np.flatnonzero(weighted_votes == best_vote)
                chosen_label = unique_labels[rng.choice(tied_indices)]

                if labels[node_index] != chosen_label:
                    labels[node_index] = int(chosen_label)
                    changed = True

            self.n_iter_ = iteration
            if not changed:
                break

        self.labels_ = labels
        self.n_communities_ = int(np.unique(labels).size)
        return self

    def fit_predict(self, adjacency_matrix):
        """
        Fit the model and return community labels.
        """
        self.fit(adjacency_matrix)
        return self.labels_.copy()
