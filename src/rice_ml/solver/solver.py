"""
Exhaustive DFS solver for Brainvita.

Finds the minimum and maximum number of pegs reachable from a given board
state, along with the best and worst first moves. Uses memoization to avoid
revisiting board states.

For large boards (27+ pegs), a budgeted mode caps the number of states
explored and returns the best solution found within that budget. When the
budget is hit, greedy rollouts are used to reach actual terminal states
(no legal moves) rather than faking a terminal at the cutoff point.
"""

import numpy as np
from .game_logic import get_valid_moves, apply_move, count_pegs, board_to_key


def _greedy_rollout_best(board):
    """
    Greedy rollout aiming to minimize pegs: at each step pick the move
    that leaves the fewest pegs after one step (greedy heuristic).
    Returns (final_peg_count, move_sequence).
    """
    current = board.copy()
    sequence = []
    while True:
        moves = get_valid_moves(current)
        if not moves:
            return count_pegs(current), sequence
        # Greedy: pick move that results in... well each move removes
        # exactly 1 peg, so just pick the first move. Instead, pick
        # the move whose resulting state has the most subsequent moves
        # (keeps options open).
        best_move = None
        best_future_moves = -1
        for move in moves:
            nb = apply_move(current, move)
            n_future = len(get_valid_moves(nb))
            if n_future > best_future_moves:
                best_future_moves = n_future
                best_move = move
        current = apply_move(current, best_move)
        sequence.append(best_move)


def _greedy_rollout_worst(board):
    """
    Greedy rollout aiming to maximize remaining pegs (get stuck fast):
    at each step pick the move whose resulting state has the fewest
    subsequent legal moves (trying to run out of moves quickly).
    Returns (final_peg_count, move_sequence).
    """
    current = board.copy()
    sequence = []
    while True:
        moves = get_valid_moves(current)
        if not moves:
            return count_pegs(current), sequence
        # Greedy: pick move that leads to fewest future moves (get stuck)
        worst_move = None
        worst_future_moves = float("inf")
        for move in moves:
            nb = apply_move(current, move)
            n_future = len(get_valid_moves(nb))
            if n_future < worst_future_moves:
                worst_future_moves = n_future
                worst_move = move
        current = apply_move(current, worst_move)
        sequence.append(worst_move)


def solve(board, max_states=None):
    """
    Solve a Brainvita board via exhaustive DFS with memoization.

    Parameters
    ----------
    board : np.ndarray
        2D board matrix (0/1/2 encoding).
    max_states : int or None
        If set, stop exploring after this many unique states and return
        the best solution found so far. None means exhaustive search.

    Returns
    -------
    dict with keys:
        - "min_pegs": int, minimum pegs reachable (best found if budgeted)
        - "max_pegs": int, maximum pegs at a true terminal state (stuck)
        - "best_move": tuple or None, the first move leading to min_pegs
        - "worst_move": tuple or None, the first move leading to max_pegs
        - "best_sequence": list of tuples, full move sequence to min_pegs
        - "worst_sequence": list of tuples, full move sequence to stuck state
        - "states_explored": int, number of unique states visited
        - "exhaustive": bool, True if search completed fully
    """
    memo = {}
    budget_hit = [False]

    def dfs(b):
        """Returns (min_pegs, min_seq, max_pegs, max_seq) for board b."""
        key = board_to_key(b)
        if key in memo:
            return memo[key]

        if max_states is not None and len(memo) >= max_states:
            budget_hit[0] = True
            # Budget exhausted — use greedy rollouts to reach actual
            # terminal states instead of faking a terminal here.
            best_pegs, best_seq = _greedy_rollout_best(b)
            worst_pegs, worst_seq = _greedy_rollout_worst(b)
            return (best_pegs, best_seq, worst_pegs, worst_seq)

        moves = get_valid_moves(b)
        if not moves:
            pegs = count_pegs(b)
            memo[key] = (pegs, [], pegs, [])
            return (pegs, [], pegs, [])

        best_pegs = float("inf")
        best_seq = []
        worst_pegs = float("-inf")
        worst_seq = []

        for move in moves:
            new_board = apply_move(b, move)
            mn, mn_seq, mx, mx_seq = dfs(new_board)
            if mn < best_pegs:
                best_pegs = mn
                best_seq = [move] + mn_seq
            if mx > worst_pegs:
                worst_pegs = mx
                worst_seq = [move] + mx_seq

        memo[key] = (best_pegs, best_seq, worst_pegs, worst_seq)
        return (best_pegs, best_seq, worst_pegs, worst_seq)

    min_pegs, best_sequence, max_pegs, worst_sequence = dfs(board)

    # Evaluate each first move to identify best and worst opening
    root_moves = get_valid_moves(board)
    best_move = None
    worst_move = None
    best_val = float("inf")
    worst_val = float("-inf")

    for move in root_moves:
        new_board = apply_move(board, move)
        mn, _, mx, _ = dfs(new_board)
        if mn < best_val:
            best_val = mn
            best_move = move
        if mx > worst_val:
            worst_val = mx
            worst_move = move

    return {
        "min_pegs": min_pegs,
        "max_pegs": max_pegs,
        "best_move": best_move,
        "worst_move": worst_move,
        "best_sequence": best_sequence,
        "worst_sequence": worst_sequence,
        "states_explored": len(memo),
        "exhaustive": not budget_hit[0],
    }
