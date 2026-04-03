"""
Deep Q-learning agent for Brainvita (from-scratch numpy MLP).

Uses a small multi-layer perceptron to approximate Q-values:

    Q(s, a) ≈ MLP(φ(s'))    where s' = apply(s, a)

Key design decisions:
    - Feature normalization is collected during a dedicated warmup phase
      using purely random play, then frozen. The replay buffer is flushed
      after freeze so no mixed-scale features persist.
    - Replay buffer stores full transitions (phi, reward, phi_next, done)
      so Q-targets are always fresh when replayed.
    - Target network (frozen copy of MLP) stabilizes training.
    - Hyperparameters can be auto-tuned via grid search.
"""

import json
import itertools
import numpy as np
from collections import deque
from ..solver.game_logic import apply_move, count_pegs, get_valid_moves
from ..data_generator.feature_extraction import extract_features


# ── From-scratch numpy MLP ──────────────────────────────────────────

class NumpyMLP:
    """Simple MLP with ReLU hidden layers and linear output."""

    def __init__(self, layer_sizes, lr=0.001):
        self.lr = lr
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        """Forward pass. Returns (output, activations_cache)."""
        activations = [x]
        current = x
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:
                current = np.maximum(0, z)
            else:
                current = z
            activations.append(current)
        return current, activations

    def predict(self, x):
        out, _ = self.forward(x)
        return float(np.squeeze(out))

    def train_step(self, x, target):
        """One SGD step. Returns loss.

        Uses saved pre-update weights for gradient propagation to
        ensure correct backpropagation through all layers.
        """
        output, activations = self.forward(x)
        target_arr = np.atleast_1d(np.asarray(target, dtype=float))
        output_arr = np.atleast_1d(output)
        loss = 0.5 * (output_arr - target_arr) ** 2
        grad = output_arr - target_arr

        # FIX #4: Compute all gradients FIRST using pre-update weights,
        # then apply all weight updates together.
        weight_grads = []
        bias_grads = []

        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = activations[i]
            dw = np.outer(a_prev.ravel(), grad.ravel())
            db = grad.ravel()
            dw = np.clip(dw, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)
            weight_grads.append((i, dw))
            bias_grads.append((i, db))
            if i > 0:
                # Use CURRENT (pre-update) weights for gradient propagation
                grad = (grad.ravel() @ self.weights[i].T)
                grad = grad * (activations[i].ravel() > 0).astype(float)

        # Now apply all updates
        for i, dw in weight_grads:
            self.weights[i] -= self.lr * dw
        for i, db in bias_grads:
            self.biases[i] -= self.lr * db

        return float(np.mean(loss))

    def train_batch(self, xs, targets):
        total_loss = 0.0
        for x, t in zip(xs, targets):
            total_loss += self.train_step(x, t)
        return total_loss / len(xs)

    def copy_from(self, other):
        """Copy weights from another MLP (for target network)."""
        self.weights = [w.copy() for w in other.weights]
        self.biases = [b.copy() for b in other.biases]

    def get_state(self):
        return {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "lr": self.lr,
        }

    def load_state(self, state):
        self.weights = [np.array(w) for w in state["weights"]]
        self.biases = [np.array(b) for b in state["biases"]]
        self.lr = state["lr"]


# ── Replay Buffer (stores full transitions) ─────────────────────────

class ReplayBuffer:
    """Stores (phi, reward, phi_next_list, done) transitions."""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, phi, reward, phi_nexts, done):
        """
        Parameters
        ----------
        phi : np.ndarray
            Feature vector of current action's resulting state.
        reward : float
            Reward received.
        phi_nexts : list of np.ndarray
            Feature vectors for all next-state actions (empty if done).
        done : bool
            Whether episode ended.
        """
        self.buffer.append((
            phi.copy(), reward,
            [p.copy() for p in phi_nexts], done
        ))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def clear(self):
        """Remove all stored transitions."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# ── Feature keys ────────────────────────────────────────────────────

FEATURE_KEYS = [
    "total_pegs", "total_empty", "total_holes", "num_legal_moves",
    "peg_ratio", "mobile_pegs", "immobile_pegs", "jumpable_pegs",
    "mobility_ratio", "num_clusters", "largest_cluster",
    "edge_pegs", "interior_pegs", "avg_adjacent_pegs",
    "avg_adjacent_empty", "max_adjacent_empty",
    "center_of_mass_r", "center_of_mass_c", "spread",
]


# ── Feature Q Agent with MLP ───────────────────────────────────────

class FeatureQAgent:
    """
    Q-learning agent with MLP function approximation.

    Parameters
    ----------
    learning_rate : float
        MLP learning rate.
    discount : float
        Discount factor (gamma).
    epsilon : float
        Initial exploration rate.
    epsilon_decay : float
        Multiply epsilon by this after each episode.
    epsilon_min : float
        Floor for epsilon.
    hidden_sizes : list of int
        Hidden layer sizes for the MLP.
    replay_capacity : int
        Experience replay buffer size.
    batch_size : int
        Minibatch size for replay.
    target_update_freq : int
        Update target network every N episodes.
    """

    def __init__(self, learning_rate=0.001, discount=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
                 hidden_sizes=None, replay_capacity=50000, batch_size=64,
                 target_update_freq=10):
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        self._hidden_sizes = hidden_sizes

        layer_sizes = [19] + hidden_sizes + [1]
        self.mlp = NumpyMLP(layer_sizes, lr=learning_rate)
        self.target_mlp = NumpyMLP(layer_sizes, lr=learning_rate)
        self.target_mlp.copy_from(self.mlp)

        # Feature normalization — frozen after warmup
        self._feat_mean = np.zeros(19)
        self._feat_std = np.ones(19)
        self._norm_frozen = False
        self._warmup_vecs = []

        self.replay = ReplayBuffer(capacity=replay_capacity)

        self.episode_rewards = []
        self.episode_pegs = []
        self.best_pegs = float("inf")
        self.best_moves = []

    def _raw_feature_vector(self, board):
        """Extract raw 19-feature vector."""
        feats = extract_features(board)
        vec = np.array([feats[k] for k in FEATURE_KEYS], dtype=float)
        np.nan_to_num(vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    def _normalize(self, vec):
        """Normalize a feature vector using current stats."""
        return (vec - self._feat_mean) / self._feat_std

    def _get_feature_vector(self, board):
        """Extract and normalize feature vector."""
        vec = self._raw_feature_vector(board)
        if not self._norm_frozen:
            self._warmup_vecs.append(vec)
        return self._normalize(vec)

    def _freeze_normalization(self):
        """Compute mean/std from warmup data, freeze, and flush replay.

        FIX #2: The replay buffer is cleared after freezing because all
        transitions stored during warmup used default (identity) normalization.
        Keeping them would mix incompatible feature scales in one training stream.
        """
        if self._warmup_vecs:
            data = np.array(self._warmup_vecs)
            self._feat_mean = np.mean(data, axis=0)
            self._feat_std = np.std(data, axis=0) + 1e-8
        self._norm_frozen = True
        self._warmup_vecs = []
        # Flush stale pre-freeze transitions
        self.replay.clear()

    def _q_value(self, phi, use_target=False):
        """Q-value from a pre-computed normalized feature vector."""
        net = self.target_mlp if use_target else self.mlp
        return net.predict(phi)

    def select_action(self, board, actions, training=True):
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return actions[np.random.randint(len(actions))]

        best_q = float("-inf")
        best_action = actions[0]
        for action in actions:
            next_board = apply_move(board, action)
            phi = self._get_feature_vector(next_board)
            q = self._q_value(phi)
            if q > best_q:
                best_q = q
                best_action = action
        return best_action

    def _replay_train(self):
        """Sample a minibatch from replay and train with fresh targets."""
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        for phi, reward, phi_nexts, done in batch:
            if done or not phi_nexts:
                target = reward
            else:
                max_q = max(self._q_value(p, use_target=True) for p in phi_nexts)
                target = reward + self.gamma * max_q
            self.mlp.train_step(phi, target)

    def _warmup_episode(self, env):
        """Run one purely random episode to collect feature stats.

        FIX #5: Warmup uses truly random action selection (no epsilon-greedy,
        no Q-value computation) to get an unbiased sample of the feature space.
        No learning occurs during warmup — only feature vectors are collected.
        """
        board = env.reset()
        move_list = []

        while not env.is_done():
            actions = env.get_actions()
            if not actions:
                break
            action = actions[np.random.randint(len(actions))]
            move_list.append(action)

            # Collect feature vector (appends to _warmup_vecs via _get_feature_vector)
            next_board = apply_move(board, action)
            self._get_feature_vector(next_board)

            board, _, _, _ = env.step(action)

        final_pegs = count_pegs(env.board)
        self.episode_pegs.append(final_pegs)
        self.episode_rewards.append(0.0)

        if final_pegs < self.best_pegs:
            self.best_pegs = final_pegs
            self.best_moves = list(move_list)

    def train_episode(self, env):
        """Run one training episode with Q-learning."""
        board = env.reset()
        total_reward = 0.0
        move_list = []

        while not env.is_done():
            actions = env.get_actions()
            if not actions:
                break

            action = self.select_action(board, actions, training=True)
            move_list.append(action)

            next_board = apply_move(board, action)
            phi = self._get_feature_vector(next_board)

            new_board, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                future_actions = env.get_actions()
                phi_nexts = []
                for fa in future_actions:
                    fb = apply_move(new_board, fa)
                    phi_nexts.append(self._get_feature_vector(fb))
            else:
                phi_nexts = []

            self.replay.add(phi, reward, phi_nexts, done)
            self._replay_train()

            board = new_board

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        final_pegs = count_pegs(env.board)
        self.episode_rewards.append(total_reward)
        self.episode_pegs.append(final_pegs)

        if final_pegs < self.best_pegs:
            self.best_pegs = final_pegs
            self.best_moves = list(move_list)

        ep_count = len(self.episode_pegs)
        if ep_count % self.target_update_freq == 0:
            self.target_mlp.copy_from(self.mlp)

        return {
            "total_reward": total_reward,
            "final_pegs": final_pegs,
            "moves_made": len(move_list),
        }

    def train(self, env, n_episodes=1000, print_every=100, warmup_episodes=50):
        """
        Train for n_episodes.

        The first warmup_episodes use purely random play (no Q-values, no
        learning) to collect feature normalization stats. After warmup,
        normalization is frozen, the replay buffer is flushed, and
        Q-learning begins with consistent feature scaling.
        """
        # Phase 1: Warmup — pure random play, collect feature stats
        if not self._norm_frozen:
            actual_warmup = min(warmup_episodes, n_episodes)
            for ep in range(1, actual_warmup + 1):
                self._warmup_episode(env)
                if ep % print_every == 0 or ep == 1:
                    recent = self.episode_pegs[-print_every:]
                    print(f"  Warmup {ep:5d} | "
                          f"avg pegs: {np.mean(recent):.1f} | "
                          f"(collecting feature stats)")
            self._freeze_normalization()
            print(f"  Normalization frozen. Replay buffer cleared. "
                  f"Starting Q-learning.\n")
            remaining = n_episodes - actual_warmup
        else:
            remaining = n_episodes

        # Phase 2: Q-learning with frozen normalization
        for ep in range(1, remaining + 1):
            self.train_episode(env)

            if ep % print_every == 0 or ep == 1:
                recent = self.episode_pegs[-print_every:]
                avg_pegs = np.mean(recent)
                min_pegs = min(recent)
                total_ep = len(self.episode_pegs)
                print(
                    f"  Episode {total_ep:5d} | "
                    f"avg pegs: {avg_pegs:.1f} | "
                    f"min pegs: {min_pegs} | "
                    f"best ever: {self.best_pegs} | "
                    f"epsilon: {self.epsilon:.3f}"
                )

        return {
            "best_pegs": self.best_pegs,
            "best_moves": self.best_moves,
            "total_episodes": len(self.episode_rewards),
        }

    def play(self, env):
        """Play one episode greedily (no exploration)."""
        board = env.reset()
        moves = []
        while not env.is_done():
            actions = env.get_actions()
            if not actions:
                break
            action = self.select_action(board, actions, training=False)
            moves.append(action)
            board, _, _, _ = env.step(action)
        return count_pegs(env.board), moves

    def save(self, path):
        data = {
            "mlp_state": self.mlp.get_state(),
            "target_mlp_state": self.target_mlp.get_state(),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "hidden_sizes": self._hidden_sizes,
            "feat_mean": self._feat_mean.tolist(),
            "feat_std": self._feat_std.tolist(),
            "norm_frozen": self._norm_frozen,
            "best_pegs": int(self.best_pegs) if np.isfinite(self.best_pegs) else None,
            "best_moves": [list(m) for m in self.best_moves],
            "episode_pegs": self.episode_pegs,
            "episode_rewards": self.episode_rewards,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Agent saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        agent = cls(
            learning_rate=data["mlp_state"]["lr"],
            discount=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_decay=data["epsilon_decay"],
            epsilon_min=data["epsilon_min"],
            hidden_sizes=data.get("hidden_sizes", [64, 32]),
            batch_size=data["batch_size"],
            target_update_freq=data.get("target_update_freq", 10),
        )
        agent.mlp.load_state(data["mlp_state"])
        if "target_mlp_state" in data:
            agent.target_mlp.load_state(data["target_mlp_state"])
        else:
            agent.target_mlp.copy_from(agent.mlp)
        agent._feat_mean = np.array(data["feat_mean"])
        agent._feat_std = np.array(data["feat_std"])
        agent._norm_frozen = data.get("norm_frozen", True)
        agent.best_pegs = data["best_pegs"] if data["best_pegs"] is not None else float("inf")
        agent.best_moves = [tuple(m) for m in data["best_moves"]]
        agent.episode_pegs = data["episode_pegs"]
        agent.episode_rewards = data["episode_rewards"]
        print(f"Agent loaded from {path} ({len(agent.episode_pegs)} episodes, "
              f"best: {agent.best_pegs} pegs)")
        return agent


# ── Hyperparameter Tuner ────────────────────────────────────────────

def tune_hyperparameters(env, trial_episodes=200, n_eval=20, verbose=True):
    """
    Grid search over hyperparameters. Each config trains for trial_episodes
    (including warmup), then evaluates with n_eval greedy episodes.

    FIX #1: Each trial agent goes through full warmup -> freeze -> train
    cycle, so train and eval use the same normalization scale.

    Returns the best config.
    """
    search_space = {
        "learning_rate": [0.0005, 0.001, 0.005],
        "discount": [0.95, 0.99],
        "epsilon_decay": [0.99, 0.995],
        "hidden_sizes": [[32, 16], [64, 32], [128, 64]],
        "batch_size": [32, 64],
        "target_update_freq": [5, 20],
    }

    keys = list(search_space.keys())
    combos = list(itertools.product(*[search_space[k] for k in keys]))

    if verbose:
        print(f"\n  HYPERPARAMETER SWEEP: {len(combos)} configs x "
              f"{trial_episodes} episodes each")
        print(f"  {'=' * 55}")

    best_avg = float("inf")
    best_config = None

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))

        hs = config["hidden_sizes"]
        if isinstance(hs, tuple):
            hs = list(hs)

        agent = FeatureQAgent(
            learning_rate=config["learning_rate"],
            discount=config["discount"],
            epsilon=1.0,
            epsilon_decay=config["epsilon_decay"],
            epsilon_min=0.05,
            hidden_sizes=hs,
            batch_size=config["batch_size"],
            target_update_freq=config["target_update_freq"],
        )

        # Full train cycle with warmup (warmup=20 for speed during tuning)
        warmup = min(20, trial_episodes // 5)
        agent.train(env, n_episodes=trial_episodes, print_every=999999,
                    warmup_episodes=warmup)

        # Evaluate with greedy play
        eval_pegs = []
        for _ in range(n_eval):
            pegs, _ = agent.play(env)
            eval_pegs.append(pegs)
        avg = np.mean(eval_pegs)

        if verbose:
            print(f"  [{i+1:3d}/{len(combos)}] "
                  f"lr={config['learning_rate']:.4f} "
                  f"gamma={config['discount']} "
                  f"eps_d={config['epsilon_decay']} "
                  f"arch={hs} "
                  f"bs={config['batch_size']} "
                  f"tuf={config['target_update_freq']} "
                  f"=> avg={avg:.1f} best={agent.best_pegs}")

        if avg < best_avg:
            best_avg = avg
            best_config = config.copy()
            best_config["hidden_sizes"] = hs

    if verbose:
        print(f"\n  BEST CONFIG (avg {best_avg:.1f} pegs):")
        for k, v in best_config.items():
            print(f"    {k}: {v}")
        print()

    return best_config
