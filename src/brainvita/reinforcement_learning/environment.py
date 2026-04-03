"""
Gym-like environment for Brainvita (Peg Solitaire).

Wraps the board state and game logic into a standard RL interface:
reset, step, get_actions, is_done.

Reward design (shaped to guide learning):
    +2.0 if mobility increases after a move (opened up options)
    -0.5 if mobility decreases (closing off options)
    +0.0 if mobility stays the same
    Terminal: -1 * final_pegs (direct penalty for remaining pegs)
"""

import numpy as np
from ..solver.game_logic import get_valid_moves, apply_move, count_pegs


class BrainvitaEnv:
    """
    Reinforcement learning environment for Brainvita.

    Parameters
    ----------
    initial_board : np.ndarray
        The starting board state (0/1/2 encoding).
    """

    def __init__(self, initial_board):
        self.initial_board = initial_board.copy()
        self.board = None
        self.starting_pegs = count_pegs(initial_board)
        self.done = False
        self.move_history = []
        self._prev_mobility = 0
        self._valid_moves_cache = None

    def reset(self):
        """Reset the board to the initial state. Returns the board."""
        self.board = self.initial_board.copy()
        self.done = False
        self.move_history = []
        self._valid_moves_cache = get_valid_moves(self.board)
        self._prev_mobility = len(self._valid_moves_cache)
        return self.board

    def get_actions(self):
        """Return list of valid moves (r1, c1, r2, c2) from current state."""
        if self.done:
            return []
        if self._valid_moves_cache is None:
            self._valid_moves_cache = get_valid_moves(self.board)
        return self._valid_moves_cache

    def step(self, action):
        """
        Take an action (move).

        Parameters
        ----------
        action : tuple
            (r1, c1, r2, c2) move to execute. Must be a legal move.

        Returns
        -------
        board : np.ndarray
            New board state.
        reward : float
            Reward for this step.
        done : bool
            True if no more moves are possible.
        info : dict
            Additional information.

        Raises
        ------
        ValueError
            If the action is not in the current set of legal moves.
        """
        # FIX #3: Validate that the action is legal before applying it
        legal_moves = self.get_actions()
        action_tuple = tuple(action)
        if action_tuple not in [tuple(m) for m in legal_moves]:
            raise ValueError(
                f"Illegal move {action_tuple}. "
                f"Legal moves: {legal_moves}"
            )

        self.board = apply_move(self.board, action)
        self.move_history.append(action_tuple)

        # Recompute valid moves and cache them
        self._valid_moves_cache = get_valid_moves(self.board)
        pegs = count_pegs(self.board)
        new_mobility = len(self._valid_moves_cache)

        if not self._valid_moves_cache:
            self.done = True
            reward = -1.0 * pegs
        else:
            mobility_delta = new_mobility - self._prev_mobility
            if mobility_delta > 0:
                reward = 2.0
            elif mobility_delta < 0:
                reward = -0.5
            else:
                reward = 0.0

        self._prev_mobility = new_mobility

        info = {
            "pegs": pegs,
            "moves_left": new_mobility,
            "total_moves_made": len(self.move_history),
        }
        return self.board, reward, self.done, info

    def is_done(self):
        """Check if the game is over."""
        return self.done

    def get_move_history(self):
        """Return the sequence of moves made so far."""
        return list(self.move_history)
