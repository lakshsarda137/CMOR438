"""Tests for rice_ml.board_constructor.board — create_board and populate_board."""

import sys
import os
import numpy as np
import pytest

# Add src to path so rice_ml is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rice_ml.board_constructor.board import create_board, populate_board


# ------------------------------------------------------------------ #
#  create_board tests                                                  #
# ------------------------------------------------------------------ #

class TestCreateBoard:
    """Tests for the create_board function."""

    def test_valid_simple_board(self):
        """A minimal 2x2 fully filled board should work."""
        mat = np.ones((2, 2), dtype=int)
        result = create_board(2, mat)
        np.testing.assert_array_equal(result, mat)

    def test_returns_copy(self):
        """Returned board must be a copy, not a reference to the input."""
        mat = np.ones((3, 3), dtype=int)
        result = create_board(3, mat)
        result[0, 0] = 99
        assert mat[0, 0] == 1

    def test_classic_cross_shape(self):
        """A cross-shaped board (like standard Brainvita) should be valid."""
        mat = np.array([
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ])
        result = create_board(7, mat)
        np.testing.assert_array_equal(result, mat)

    def test_l_shaped_board(self):
        """An L-shaped board where every 1 has at least one neighbor."""
        mat = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ])
        result = create_board(3, mat)
        np.testing.assert_array_equal(result, mat)

    def test_isolated_circle_raises(self):
        """A circle with no adjacent circles must raise ValueError."""
        mat = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ])
        with pytest.raises(ValueError, match="isolated"):
            create_board(3, mat)

    def test_single_circle_raises(self):
        """A board with only one circle should fail (need at least 2)."""
        mat = np.array([
            [0, 0],
            [0, 1],
        ])
        with pytest.raises(ValueError, match="at least 2"):
            create_board(2, mat)

    def test_wrong_shape_raises(self):
        """Matrix shape must match total_circles."""
        mat = np.ones((3, 3), dtype=int)
        with pytest.raises(ValueError, match="shape"):
            create_board(4, mat)

    def test_invalid_values_raises(self):
        """Matrix entries other than 0 and 1 must raise."""
        mat = np.array([[1, 2], [1, 1]])
        with pytest.raises(ValueError, match="only 0s and 1s"):
            create_board(2, mat)

    def test_non_positive_total_circles_raises(self):
        """total_circles must be a positive integer."""
        with pytest.raises(ValueError, match="positive integer"):
            create_board(0, np.array([]))

    def test_negative_total_circles_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            create_board(-1, np.array([]))


# ------------------------------------------------------------------ #
#  populate_board tests                                                #
# ------------------------------------------------------------------ #

class TestPopulateBoard:
    """Tests for the populate_board function."""

    @pytest.fixture
    def cross_board(self):
        """Standard 7x7 cross-shaped board."""
        return np.array([
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ])

    def test_single_empty(self, cross_board):
        """One empty position in the center — classic Brainvita start."""
        board = populate_board(cross_board, 1, [(3, 3)])
        assert board[3, 3] == 1  # empty hole
        assert board[0, 2] == 2  # peg
        assert board[0, 0] == 0  # no circle

    def test_multiple_empties(self, cross_board):
        """Multiple empty positions should all become 1."""
        empties = [(3, 3), (2, 2), (4, 4)]
        board = populate_board(cross_board, 3, empties)
        for r, c in empties:
            assert board[r, c] == 1
        # Non-empty circles should be 2
        assert board[0, 2] == 2

    def test_zero_encoding(self, cross_board):
        """Positions with no circle must remain 0."""
        board = populate_board(cross_board, 1, [(3, 3)])
        assert board[0, 0] == 0
        assert board[0, 1] == 0

    def test_all_values_in_range(self, cross_board):
        """All values in the populated board must be 0, 1, or 2."""
        board = populate_board(cross_board, 2, [(3, 3), (0, 3)])
        assert set(np.unique(board)).issubset({0, 1, 2})

    def test_peg_count(self, cross_board):
        """Number of pegs should equal total circles minus empty circles."""
        n_empty = 3
        empties = [(3, 3), (2, 0), (5, 3)]
        board = populate_board(cross_board, n_empty, empties)
        total_circles = int(cross_board.sum())
        n_pegs = int((board == 2).sum())
        assert n_pegs == total_circles - n_empty

    def test_empty_on_nonexistent_circle_raises(self):
        """Placing an empty on a 0-entry must raise."""
        mat = np.array([[1, 1], [1, 0]])
        with pytest.raises(ValueError, match="does not correspond"):
            populate_board(mat, 1, [(1, 1)])

    def test_count_mismatch_raises(self):
        """empty_circles must match len(empty_positions)."""
        mat = np.ones((2, 2), dtype=int)
        with pytest.raises(ValueError, match="does not match"):
            populate_board(mat, 2, [(0, 0)])

    def test_too_many_empties_raises(self):
        """Cannot have more empties than circles."""
        mat = np.array([[1, 1], [0, 0]])
        with pytest.raises(ValueError, match="exceeds"):
            populate_board(mat, 3, [(0, 0), (0, 1), (0, 0)])

    def test_duplicate_positions_raises(self):
        """Duplicate empty positions must raise."""
        mat = np.ones((3, 3), dtype=int)
        with pytest.raises(ValueError, match="Duplicate"):
            populate_board(mat, 2, [(1, 1), (1, 1)])

    def test_out_of_bounds_raises(self):
        """Positions outside the matrix must raise."""
        mat = np.ones((2, 2), dtype=int)
        with pytest.raises(ValueError, match="out of bounds"):
            populate_board(mat, 1, [(5, 5)])

    def test_small_board_full_populate(self):
        """A 2x2 board with zero empties — all pegs."""
        mat = np.ones((2, 2), dtype=int)
        board = populate_board(mat, 0, [])
        expected = np.full((2, 2), 2)
        np.testing.assert_array_equal(board, expected)
