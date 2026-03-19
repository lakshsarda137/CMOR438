"""
Board construction and population for Brainvita (Peg Solitaire).

Provides functions to create arbitrary board shapes and populate them
with pegs, supporting any orientation, size, and initial empty positions.
"""

import numpy as np


def create_board(total_circles, circle_existence_matrix):
    """
    Create a raw Brainvita board with no balls placed.

    Parameters
    ----------
    total_circles : int
        The side length of the square grid. The board is represented
        as a total_circles x total_circles matrix.
    circle_existence_matrix : np.ndarray
        A (total_circles x total_circles) binary matrix where 1 indicates
        a circle exists at that position and 0 indicates it does not.
        Every 1-entry must have at least one adjacent 1-entry (up, down,
        left, or right) — isolated circles are forbidden.

    Returns
    -------
    np.ndarray
        A copy of the validated circle_existence_matrix.

    Raises
    ------
    ValueError
        If total_circles is not a positive integer.
    ValueError
        If circle_existence_matrix shape does not match (total_circles, total_circles).
    ValueError
        If circle_existence_matrix contains values other than 0 and 1.
    ValueError
        If any 1-entry has no adjacent 1-entry.
    ValueError
        If the board has fewer than 2 circles (no game possible).
    """
    if not isinstance(total_circles, (int, np.integer)) or total_circles <= 0:
        raise ValueError("total_circles must be a positive integer.")

    matrix = np.asarray(circle_existence_matrix, dtype=int)

    if matrix.shape != (total_circles, total_circles):
        raise ValueError(
            f"circle_existence_matrix shape {matrix.shape} does not match "
            f"({total_circles}, {total_circles})."
        )

    if not np.isin(matrix, [0, 1]).all():
        raise ValueError("circle_existence_matrix must contain only 0s and 1s.")

    ones = np.argwhere(matrix == 1)

    if len(ones) < 2:
        raise ValueError("Board must have at least 2 circles.")

    for r, c in ones:
        neighbors = []
        if r > 0:
            neighbors.append(matrix[r - 1, c])
        if r < total_circles - 1:
            neighbors.append(matrix[r + 1, c])
        if c > 0:
            neighbors.append(matrix[r, c - 1])
        if c < total_circles - 1:
            neighbors.append(matrix[r, c + 1])

        if sum(neighbors) == 0:
            raise ValueError(
                f"Circle at ({r}, {c}) is isolated — every circle must have "
                f"at least one adjacent circle."
            )

    return matrix.copy()


def populate_board(circle_existence_matrix, empty_circles, empty_positions):
    """
    Populate a board with balls, leaving specified positions empty.

    Parameters
    ----------
    circle_existence_matrix : np.ndarray
        A binary matrix from create_board (1 = circle exists, 0 = no circle).
    empty_circles : int
        The number of circles to leave empty (without a ball).
    empty_positions : list of tuple
        A list of (row, col) tuples indicating which circles should be empty.
        Each position must correspond to a 1 in circle_existence_matrix.
        Length must equal empty_circles.

    Returns
    -------
    np.ndarray
        A matrix of the same shape where:
        - 0 = no circle at this position
        - 1 = circle exists but has no ball (empty hole)
        - 2 = circle exists and has a ball (peg)

    Raises
    ------
    ValueError
        If empty_circles does not match len(empty_positions).
    ValueError
        If any empty_position falls on a 0-entry in circle_existence_matrix.
    ValueError
        If empty_circles exceeds the total number of circles on the board.
    ValueError
        If empty_positions contains duplicates.
    """
    matrix = np.asarray(circle_existence_matrix, dtype=int)

    if not isinstance(empty_circles, (int, np.integer)) or empty_circles < 0:
        raise ValueError("empty_circles must be a non-negative integer.")

    if len(empty_positions) != empty_circles:
        raise ValueError(
            f"empty_circles ({empty_circles}) does not match the number of "
            f"empty_positions provided ({len(empty_positions)})."
        )

    total_circles_on_board = int(matrix.sum())

    if empty_circles > total_circles_on_board:
        raise ValueError(
            f"empty_circles ({empty_circles}) exceeds total circles on "
            f"board ({total_circles_on_board})."
        )

    # Check for duplicate positions
    pos_set = set()
    for pos in empty_positions:
        if pos in pos_set:
            raise ValueError(f"Duplicate empty position: {pos}.")
        pos_set.add(pos)

    # Validate each empty position sits on an existing circle
    for r, c in empty_positions:
        if r < 0 or r >= matrix.shape[0] or c < 0 or c >= matrix.shape[1]:
            raise ValueError(
                f"Empty position ({r}, {c}) is out of bounds."
            )
        if matrix[r, c] != 1:
            raise ValueError(
                f"Empty position ({r}, {c}) does not correspond to an "
                f"existing circle in circle_existence_matrix."
            )

    # Build populated board: 0 stays 0, circles become 2 (ball), empties become 1
    board = matrix * 2  # all circles get balls

    for r, c in empty_positions:
        board[r, c] = 1  # remove ball, keep circle

    return board
