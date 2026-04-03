"""
Brainvita (Peg Solitaire) game logic.

A move consists of jumping a peg over an adjacent peg (horizontally or
vertically) into an empty hole. The jumped peg is removed.

Board encoding:
    0 = no hole at this position
    1 = empty hole
    2 = hole with a peg
"""

import numpy as np


def get_valid_moves(board):
    """
    Find all valid moves on the current board.

    A valid move is a tuple (r1, c1, r2, c2) meaning: the peg at (r1, c1)
    jumps over the peg at the midpoint and lands on the empty hole at (r2, c2).

    Parameters
    ----------
    board : np.ndarray
        2D board matrix (0/1/2 encoding).

    Returns
    -------
    list of tuple
        Each tuple is (from_row, from_col, to_row, to_col).
    """
    rows, cols = board.shape
    moves = []
    # Directions: (row_delta, col_delta) for the jump destination
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    for r in range(rows):
        for c in range(cols):
            if board[r, c] != 2:
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                mr, mc = r + dr // 2, c + dc // 2  # midpoint
                if 0 <= nr < rows and 0 <= nc < cols:
                    if board[mr, mc] == 2 and board[nr, nc] == 1:
                        moves.append((r, c, nr, nc))
    return moves


def apply_move(board, move):
    """
    Apply a move and return the new board state.

    Parameters
    ----------
    board : np.ndarray
        2D board matrix.
    move : tuple
        (from_row, from_col, to_row, to_col).

    Returns
    -------
    np.ndarray
        New board after the move.
    """
    r1, c1, r2, c2 = move
    mr, mc = (r1 + r2) // 2, (c1 + c2) // 2

    new_board = board.copy()
    new_board[r1, c1] = 1  # peg leaves
    new_board[mr, mc] = 1  # jumped peg removed
    new_board[r2, c2] = 2  # peg lands
    return new_board


def count_pegs(board):
    """Return the number of pegs on the board."""
    return int(np.sum(board == 2))


def board_to_key(board):
    """Convert board to a hashable key for memoization."""
    return board.tobytes()
