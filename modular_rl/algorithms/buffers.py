import random
import numpy as np
import torch
from collections import deque
from typing import List, Optional, Tuple, Union


# ══════════════════════════════════════════════════════════════════════════════
# Standard Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Standard i.i.d. replay buffer for off-policy agents (DQN, SAC, TD3).
    Stores individual transitions and samples uniformly at random.
    """

    def __init__(self, capacity: int, device: str = "cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = torch.device(device)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state_np = np.array(state, dtype=np.float32)
        next_state_np = np.array(next_state, dtype=np.float32)
        action_np = (
            np.array(action, dtype=np.int64)
            if isinstance(action[0], (int, np.integer))
            else np.array(action, dtype=np.float32)
        )
        reward_np = np.array(reward, dtype=np.float32)
        done_np = np.array(done, dtype=np.float32)

        return (
            torch.from_numpy(state_np).to(self.device),
            torch.from_numpy(action_np).to(self.device),
            torch.from_numpy(reward_np).unsqueeze(1).to(self.device),
            torch.from_numpy(next_state_np).to(self.device),
            torch.from_numpy(done_np).unsqueeze(1).to(self.device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# Prioritized Replay Buffer (PER)
# ══════════════════════════════════════════════════════════════════════════════

class _SumTree:
    """Binary sum-tree for O(log n) priority-weighted sampling and updates."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.n_entries = 0
        self._write = 0

    # -- internal helpers --------------------------------------------------

    def _propagate(self, leaf_idx: int, delta: float):
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def _leaf_idx(self, data_idx: int) -> int:
        return data_idx + self.capacity - 1

    # -- public API --------------------------------------------------------

    def add(self, priority: float, data):
        leaf = self._leaf_idx(self._write)
        self.update(leaf, priority)
        self.data[self._write] = data
        self._write = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float):
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def retrieve(self, value: float) -> Tuple[int, float, object]:
        """Returns (leaf_idx, priority, data)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (PER).

    Transitions with higher TD error are sampled more frequently.
    Importance-sampling (IS) weights correct the resulting bias.

    Parameters
    ----------
    capacity          : Maximum number of transitions.
    alpha             : Priority exponent (0 = uniform, 1 = fully prioritized).
    beta_start        : Initial IS exponent (corrects for non-uniform sampling).
    beta_end          : Final IS exponent (annealed to 1.0 during training).
    beta_decay_steps  : Steps over which beta is linearly annealed.
    epsilon           : Small constant added to priorities to prevent zero.

    Usage
    -----
    buffer = PrioritizedReplayBuffer(capacity=10000, device="cpu")
    # Use with DQNAgent — PER is detected automatically and IS weights are applied.
    """

    def __init__(
        self,
        capacity: int,
        device: str = "cpu",
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_decay_steps: int = 100_000,
        epsilon: float = 1e-6,
    ):
        self._tree = _SumTree(capacity)
        self.device = torch.device(device)
        self.alpha = alpha
        self._beta = beta_start
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._beta_decay = beta_decay_steps
        self._epsilon = epsilon
        self._max_priority = 1.0
        self._step = 0

    def push(self, state, action, reward, next_state, done):
        self._tree.add(self._max_priority ** self.alpha, (state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Returns the standard 5 tensors PLUS:
            weights : (B, 1)  — importance-sampling correction weights
            indices : list[int] — leaf indices for priority updates
        """
        self._step += 1
        frac = min(1.0, self._step / self._beta_decay)
        self._beta = self._beta_start + frac * (self._beta_end - self._beta_start)

        total = self._tree.total
        segment = total / batch_size

        idxs, priorities, batch = [], [], []
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self._tree.retrieve(s)
            if data is None:          # guard against uninitialised slots
                idx, priority, data = self._tree.retrieve(random.uniform(0, total))
            idxs.append(idx)
            priorities.append(max(priority, self._epsilon))
            batch.append(data)

        probs = np.array(priorities, dtype=np.float32) / (total + 1e-8)
        weights = (self._tree.n_entries * probs) ** (-self._beta)
        weights = (weights / weights.max()).astype(np.float32)

        state, action, reward, next_state, done = zip(*batch)
        state_np = np.array(state, dtype=np.float32)
        next_state_np = np.array(next_state, dtype=np.float32)
        action_np = (
            np.array(action, dtype=np.int64)
            if isinstance(action[0], (int, np.integer))
            else np.array(action, dtype=np.float32)
        )
        reward_np = np.array(reward, dtype=np.float32)
        done_np = np.array(done, dtype=np.float32)

        return (
            torch.from_numpy(state_np).to(self.device),
            torch.from_numpy(action_np).to(self.device),
            torch.from_numpy(reward_np).unsqueeze(1).to(self.device),
            torch.from_numpy(next_state_np).to(self.device),
            torch.from_numpy(done_np).unsqueeze(1).to(self.device),
            torch.from_numpy(weights).unsqueeze(1).to(self.device),
            idxs,
        )

    def update_priorities(self, idxs: List[int], td_errors: np.ndarray):
        """Update priorities after computing TD errors."""
        priorities = (np.abs(td_errors) + self._epsilon) ** self.alpha
        for idx, priority in zip(idxs, priorities):
            self._tree.update(idx, float(priority))
            self._max_priority = max(self._max_priority, float(priority))

    def __len__(self) -> int:
        return self._tree.n_entries


# ══════════════════════════════════════════════════════════════════════════════
# Sequence Replay Buffer (DRQN / Transformer)
# ══════════════════════════════════════════════════════════════════════════════

class SequenceReplayBuffer:
    """
    Episode-based replay buffer for recurrent agents (DRQN, Transformer).
    Stores complete episodes and samples sub-sequences of length L with padding.
    """

    def __init__(self, capacity_episodes: int, device: str = "cpu"):
        self.buffer = deque(maxlen=capacity_episodes)
        self.current_episode: list = []
        self.device = torch.device(device)

    def push(self, state, action, reward, next_state, done):
        self.current_episode.append((state, action, reward, next_state, done))
        if done:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, ...]:
        episodes = random.sample(self.buffer, batch_size)

        batch_s, batch_a, batch_r, batch_ns, batch_d, batch_m = [], [], [], [], [], []

        for episode in episodes:
            ep_len = len(episode)
            if ep_len <= seq_len:
                start_idx, slice_len = 0, ep_len
            else:
                start_idx = random.randint(0, ep_len - seq_len)
                slice_len = seq_len

            sub = episode[start_idx: start_idx + slice_len]
            states, actions, rewards, next_states, dones = zip(*sub)

            states = list(states)
            actions = list(actions)
            rewards = list(rewards)
            next_states = list(next_states)
            dones = list(dones)
            masks = [1.0] * slice_len

            pad = seq_len - slice_len
            if pad > 0:
                state_shape = sub[0][0].shape
                a_shape = np.array(sub[0][1]).shape
                states.extend([np.zeros(state_shape, dtype=np.float32)] * pad)
                next_states.extend([np.zeros(state_shape, dtype=np.float32)] * pad)
                if isinstance(sub[0][1], (int, np.integer)):
                    actions.extend([0] * pad)
                else:
                    actions.extend([np.zeros(a_shape, dtype=np.float32)] * pad)
                rewards.extend([0.0] * pad)
                dones.extend([True] * pad)
                masks.extend([0.0] * pad)

            batch_s.append(states)
            batch_a.append(actions)
            batch_r.append(rewards)
            batch_ns.append(next_states)
            batch_d.append(dones)
            batch_m.append(masks)

        s_np = np.array(batch_s, dtype=np.float32)
        ns_np = np.array(batch_ns, dtype=np.float32)
        a_np = (
            np.array(batch_a, dtype=np.int64)
            if isinstance(episodes[0][0][1], (int, np.integer))
            else np.array(batch_a, dtype=np.float32)
        )
        r_np = np.array(batch_r, dtype=np.float32)
        d_np = np.array(batch_d, dtype=np.float32)
        m_np = np.array(batch_m, dtype=np.float32)

        return (
            torch.from_numpy(s_np).to(self.device),
            torch.from_numpy(a_np).to(self.device),
            torch.from_numpy(r_np).unsqueeze(-1).to(self.device),
            torch.from_numpy(ns_np).to(self.device),
            torch.from_numpy(d_np).unsqueeze(-1).to(self.device),
            torch.from_numpy(m_np).unsqueeze(-1).to(self.device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# Rollout Buffer (PPO)
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Fixed-length on-policy rollout buffer for PPO.
    Stores (state, action, reward, value, log_prob, done) for N steps,
    then computes GAE advantages and yields mini-batches for K epochs of updates.

    Parameters
    ----------
    n_steps     : Rollout length (number of environment steps per update).
    gamma       : Discount factor.
    gae_lambda  : GAE lambda (0 = TD(0), 1 = Monte-Carlo).
    """

    def __init__(
        self,
        n_steps: int,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.n_steps = n_steps
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._states: list = []
        self._actions: list = []
        self._rewards: list = []
        self._values: list = []
        self._log_probs: list = []
        self._dones: list = []

    def push(self, state, action, reward: float, value: float, log_prob: float, done: bool):
        self._states.append(np.array(state, dtype=np.float32))
        self._actions.append(action)
        self._rewards.append(float(reward))
        self._values.append(float(value))
        self._log_probs.append(float(log_prob))
        self._dones.append(float(done))

    def is_full(self) -> bool:
        return len(self._states) >= self.n_steps

    def _compute_gae(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self._rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self._values[t + 1]
            non_terminal = 1.0 - self._dones[t]
            delta = self._rewards[t] + self.gamma * next_val * non_terminal - self._values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
        returns = advantages + np.array(self._values, dtype=np.float32)
        return advantages, returns

    def get_batches(self, batch_size: int, last_value: float = 0.0):
        """
        Yield (states, actions, old_log_probs, advantages, returns) mini-batches.
        Advantages are normalised per rollout.
        """
        advantages, returns = self._compute_gae(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.from_numpy(np.stack(self._states)).to(self.device)
        if isinstance(self._actions[0], (int, np.integer)):
            actions = torch.tensor(self._actions, dtype=torch.long).to(self.device)
        else:
            actions = torch.from_numpy(np.array(self._actions, dtype=np.float32)).to(self.device)
        log_probs = torch.tensor(self._log_probs, dtype=torch.float32).to(self.device)
        adv_t = torch.from_numpy(advantages).to(self.device)
        ret_t = torch.from_numpy(returns).to(self.device)

        n = len(self._states)
        for idx in [
            torch.from_numpy(p).to(self.device)
            for p in np.array_split(np.random.permutation(n), max(1, n // batch_size))
        ]:
            yield states[idx], actions[idx], log_probs[idx], adv_t[idx], ret_t[idx]

    def reset(self):
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._values.clear()
        self._log_probs.clear()
        self._dones.clear()

    def __len__(self) -> int:
        return len(self._states)
