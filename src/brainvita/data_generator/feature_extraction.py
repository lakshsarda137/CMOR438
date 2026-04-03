"""
Feature extraction from Brainvita board states.

Computes the feature vector for a given populated board (0/1/2 encoding)
that will be used as input to supervised learning models.
"""

import numpy as np
from ..solver.game_logic import get_valid_moves


def _find_clusters(board):
    """
    Find connected components of pegs using BFS.
    Returns list of cluster sizes.
    """
    rows, cols = board.shape
    visited = np.zeros_like(board, dtype=bool)
    clusters = []

    for r in range(rows):
        for c in range(cols):
            if board[r, c] != 2 or visited[r, c]:
                continue
            # BFS from this peg
            queue = [(r, c)]
            visited[r, c] = True
            size = 0
            while queue:
                cr, cc = queue.pop(0)
                size += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and board[nr, nc] == 2 and not visited[nr, nc]):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            clusters.append(size)

    return clusters


def extract_features(board):
    """
    Extract feature dictionary from a populated board.

    Parameters
    ----------
    board : np.ndarray
        2D board matrix (0=no hole, 1=empty hole, 2=peg).

    Returns
    -------
    dict
        Feature name -> value.
    """
    rows, cols = board.shape

    peg_positions = list(zip(*np.where(board == 2)))
    empty_positions = list(zip(*np.where(board == 1)))
    hole_positions = list(zip(*np.where(board >= 1)))

    total_pegs = len(peg_positions)
    total_empty = len(empty_positions)
    total_holes = len(hole_positions)

    legal_moves = get_valid_moves(board)
    num_legal_moves = len(legal_moves)

    peg_ratio = total_pegs / total_holes if total_holes > 0 else 0.0

    # Mobility: which pegs can move, which can be jumped over
    mobile_set = set()
    jumpable_set = set()
    for r1, c1, r2, c2 in legal_moves:
        mobile_set.add((r1, c1))
        mr, mc = (r1 + r2) // 2, (c1 + c2) // 2
        jumpable_set.add((mr, mc))

    mobile_pegs = len(mobile_set)
    immobile_pegs = total_pegs - mobile_pegs
    jumpable_pegs = len(jumpable_set)
    mobility_ratio = mobile_pegs / total_pegs if total_pegs > 0 else 0.0

    # Clusters
    clusters = _find_clusters(board)
    num_clusters = len(clusters)
    largest_cluster = max(clusters) if clusters else 0

    # Edge vs interior pegs
    # A peg is on the "edge" if at least one of its 4 neighbors is not a hole
    hole_set = set(hole_positions)
    edge_pegs = 0
    interior_pegs = 0
    for r, c in peg_positions:
        on_edge = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in hole_set:
                on_edge = True
                break
        if on_edge:
            edge_pegs += 1
        else:
            interior_pegs += 1

    # Adjacency stats
    adj_pegs_list = []
    adj_empty_list = []
    max_adj_empty = 0
    for r, c in peg_positions:
        ap = 0
        ae = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if board[nr, nc] == 2:
                    ap += 1
                elif board[nr, nc] == 1:
                    ae += 1
        adj_pegs_list.append(ap)
        adj_empty_list.append(ae)
        if ae > max_adj_empty:
            max_adj_empty = ae

    avg_adjacent_pegs = np.mean(adj_pegs_list) if adj_pegs_list else 0.0
    avg_adjacent_empty = np.mean(adj_empty_list) if adj_empty_list else 0.0

    # Center of mass and spread
    if peg_positions:
        peg_arr = np.array(peg_positions, dtype=float)
        com_r = np.mean(peg_arr[:, 0])
        com_c = np.mean(peg_arr[:, 1])
        spread = np.std(np.sqrt((peg_arr[:, 0] - com_r) ** 2
                                + (peg_arr[:, 1] - com_c) ** 2))
    else:
        com_r = com_c = spread = 0.0

    return {
        "total_pegs": total_pegs,
        "total_empty": total_empty,
        "total_holes": total_holes,
        "num_legal_moves": num_legal_moves,
        "peg_ratio": round(peg_ratio, 6),
        "mobile_pegs": mobile_pegs,
        "immobile_pegs": immobile_pegs,
        "jumpable_pegs": jumpable_pegs,
        "mobility_ratio": round(mobility_ratio, 6),
        "num_clusters": num_clusters,
        "largest_cluster": largest_cluster,
        "edge_pegs": edge_pegs,
        "interior_pegs": interior_pegs,
        "avg_adjacent_pegs": round(avg_adjacent_pegs, 6),
        "avg_adjacent_empty": round(avg_adjacent_empty, 6),
        "max_adjacent_empty": max_adj_empty,
        "center_of_mass_r": round(com_r, 6),
        "center_of_mass_c": round(com_c, 6),
        "spread": round(spread, 6),
    }
