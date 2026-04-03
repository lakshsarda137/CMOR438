"""
Visualization utilities for Brainvita boards.

Renders boards produced by create_board and populate_board as
matplotlib figures with circles representing holes and filled
circles representing pegs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_board(board_matrix, title="Brainvita Board"):
    """
    Visualize a board matrix.

    Handles both raw boards (from create_board, values 0/1) and
    populated boards (from populate_board, values 0/1/2).

    Parameters
    ----------
    board_matrix : np.ndarray
        A 2D matrix where:
        - 0 = no circle
        - 1 = empty circle (hole)
        - 2 = circle with ball (peg)
        If the matrix contains only 0s and 1s (raw board), all
        circles are drawn as empty holes.
    title : str
        Title for the plot.
    """
    matrix = np.asarray(board_matrix, dtype=int)
    rows, cols = matrix.shape
    is_populated = np.any(matrix == 2)

    fig, ax = plt.subplots(1, 1, figsize=(max(cols, 4), max(rows, 4)))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, alpha=0.2)

    radius = 0.35

    for r in range(rows):
        for c in range(cols):
            val = matrix[r, c]
            if val == 0:
                continue
            elif val == 1:
                if is_populated:
                    # Empty hole on a populated board
                    circle = patches.Circle(
                        (c, r), radius,
                        facecolor="white", edgecolor="black", linewidth=2
                    )
                else:
                    # Raw board — just show the hole exists
                    circle = patches.Circle(
                        (c, r), radius,
                        facecolor="lightyellow", edgecolor="gray", linewidth=1.5
                    )
                ax.add_patch(circle)
            elif val == 2:
                # Peg
                circle = patches.Circle(
                    (c, r), radius,
                    facecolor="steelblue", edgecolor="darkblue", linewidth=2
                )
                ax.add_patch(circle)

    ax.set_facecolor("burlywood")
    fig.tight_layout()
    plt.show()
