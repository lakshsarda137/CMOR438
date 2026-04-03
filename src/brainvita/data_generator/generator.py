"""
Synthetic data generator for Brainvita.

Generates unique random board configurations, solves each one exhaustively,
extracts features, and exports labeled datasets as CSV for supervised learning.

Supports parallel processing via multiprocessing for faster generation.
"""

import csv
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

from ..board_constructor.board import populate_board
from ..solver.solver import solve
from ..solver.game_logic import count_pegs, board_to_key
from .feature_extraction import extract_features


def generate_random_board_shape(num_circles, grid_size=None):
    """
    Generate a random connected board shape with exactly num_circles holes.

    Grows from a seed by randomly adding adjacent cells, guaranteeing
    that every cell has at least one neighbor (connectivity).

    Parameters
    ----------
    num_circles : int
        Number of circles in the board.
    grid_size : int or None
        Side length of the underlying grid. If None, auto-computed.

    Returns
    -------
    np.ndarray
        A trimmed binary matrix (1 = circle, 0 = no circle).
    """
    if grid_size is None:
        grid_size = max(int(np.sqrt(num_circles)) + 3, 5)

    matrix = np.zeros((grid_size, grid_size), dtype=int)
    center = grid_size // 2
    matrix[center, center] = 1
    active = [(center, center)]
    count = 1

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while count < num_circles:
        idx = np.random.randint(len(active))
        r, c = active[idx]
        np.random.shuffle(directions)
        grown = False
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and matrix[nr, nc] == 0:
                matrix[nr, nc] = 1
                active.append((nr, nc))
                count += 1
                grown = True
                break
        if not grown:
            active.pop(idx)
            if not active:
                break

    rows = np.any(matrix == 1, axis=1)
    cols = np.any(matrix == 1, axis=0)
    matrix = matrix[rows][:, cols]
    return matrix


def generate_random_config(num_circles, min_empty=1, max_empty=3):
    """
    Generate a single random board configuration with a specific circle count.

    Returns
    -------
    np.ndarray
        Populated board (0/1/2 encoding).
    """
    board_shape = generate_random_board_shape(num_circles)
    actual_circles = int(board_shape.sum())

    max_e = min(max_empty, actual_circles - 1)
    min_e = min(min_empty, max_e)
    n_empty = np.random.randint(min_e, max_e + 1)

    circle_positions = list(zip(*np.where(board_shape == 1)))
    chosen = np.random.choice(len(circle_positions), size=n_empty, replace=False)
    empty_positions = [circle_positions[i] for i in chosen]

    populated = populate_board(board_shape, n_empty, empty_positions)
    return populated


def _process_single_board(args):
    """
    Worker function for parallel processing.
    Generates a board, solves it, extracts features.
    Returns a dict row or None if it fails.
    """
    num_circles, min_empty, max_empty, seed = args
    np.random.seed(seed)

    populated = generate_random_config(num_circles, min_empty, max_empty)

    try:
        solution = solve(populated)
    except Exception:
        return None

    # Skip boards where no moves are possible (useless for training)
    if solution["best_move"] is None:
        return None

    features = extract_features(populated)

    best = solution["best_move"]
    worst = solution["worst_move"]

    row = {
        **features,
        "min_pegs_reachable": solution["min_pegs"],
        "max_pegs_reachable": solution["max_pegs"],
        "best_move_r1": best[0] if best else -1,
        "best_move_c1": best[1] if best else -1,
        "best_move_r2": best[2] if best else -1,
        "best_move_c2": best[3] if best else -1,
        "worst_move_r1": worst[0] if worst else -1,
        "worst_move_c1": worst[1] if worst else -1,
        "worst_move_r2": worst[2] if worst else -1,
        "worst_move_c2": worst[3] if worst else -1,
        "board_rows": populated.shape[0],
        "board_cols": populated.shape[1],
        "board_state": populated.tobytes().hex(),
    }
    return row


def generate_csv(output_path, samples_per_size=500, min_circles=6,
                 max_circles=25, min_empty=1, max_empty=3,
                 n_workers=None, seed=42):
    """
    Generate a full CSV dataset with parallel processing.

    Parameters
    ----------
    output_path : str
        Path to the output CSV file.
    samples_per_size : int
        Number of boards to generate per circle count.
    min_circles : int
        Smallest board size.
    max_circles : int
        Largest board size.
    min_empty : int
        Minimum empty holes per board.
    max_empty : int
        Maximum empty holes per board.
    n_workers : int or None
        Number of parallel workers. None = all CPU cores.
    seed : int
        Base random seed (each task gets seed + offset for reproducibility).

    Returns
    -------
    int
        Number of rows written.
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Build task list: (num_circles, min_empty, max_empty, seed)
    tasks = []
    task_seed = seed
    for n_circles in range(min_circles, max_circles + 1):
        for _ in range(samples_per_size):
            tasks.append((n_circles, min_empty, max_empty, task_seed))
            task_seed += 1

    total_tasks = len(tasks)
    print(f"Generating {total_tasks} boards ({min_circles}-{max_circles} circles, "
          f"{samples_per_size}/size)")
    print(f"Using {n_workers} workers")

    fieldnames = [
        "total_pegs", "total_empty", "total_holes", "num_legal_moves",
        "peg_ratio", "mobile_pegs", "immobile_pegs", "jumpable_pegs",
        "mobility_ratio", "num_clusters", "largest_cluster",
        "edge_pegs", "interior_pegs", "avg_adjacent_pegs",
        "avg_adjacent_empty", "max_adjacent_empty",
        "center_of_mass_r", "center_of_mass_c", "spread",
        "min_pegs_reachable", "max_pegs_reachable",
        "best_move_r1", "best_move_c1", "best_move_r2", "best_move_c2",
        "worst_move_r1", "worst_move_c1", "worst_move_r2", "worst_move_c2",
        "board_rows", "board_cols", "board_state",
    ]

    rows_written = 0
    seen_keys = set()
    t0 = time.time()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_process_single_board,
                                                           tasks, chunksize=10)):
                if result is None:
                    continue

                # Deduplicate by board_state
                state_key = result["board_state"]
                if state_key in seen_keys:
                    continue
                seen_keys.add(state_key)

                writer.writerow(result)
                rows_written += 1

                if rows_written % 100 == 0:
                    f.flush()
                    elapsed = time.time() - t0
                    rate = rows_written / elapsed
                    print(f"  {rows_written} rows written | "
                          f"{elapsed:.0f}s elapsed | "
                          f"{rate:.1f} rows/s | "
                          f"task {i+1}/{total_tasks}")

    elapsed = time.time() - t0
    print(f"\nDone: {rows_written} unique rows in {elapsed:.1f}s")
    print(f"Saved to {output_path}")
    return rows_written
