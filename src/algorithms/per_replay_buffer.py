"""
Prioritized Experience Replay (PER) Buffer for QMIX.

Instead of uniform random sampling, transitions with higher TD error
are sampled more often — so rare "success" experiences get replayed
more frequently.

Reference: Schaul et al., "Prioritized Experience Replay", ICLR 2016.
"""

import numpy as np
import torch


class SumTree:
    """
    Binary tree where each leaf stores a priority, and each internal node
    stores the sum of its children. Enables O(log n) priority sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        idx = min(idx, 2 * self.capacity - 1)
        data_idx = max(0, min(idx - self.capacity, self.capacity - 1))
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self):
        return self.n_entries


class PrioritizedReplayBuffer:
    """
    PER buffer: samples high-TD-error transitions more frequently.

    Key params:
        alpha: how much prioritization (0 = uniform, 1 = full priority)
        beta:  importance sampling correction (annealed 0.4 → 1.0)
        eps:   small constant to avoid zero priority
    """

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        total_steps: int = 2_000_000,
        eps: float = 1e-6,
    ):
        self.tree = SumTree(capacity)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps
        self.eps = eps
        self.max_priority = 1.0

    def push(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        state,
        next_state,
    ):
        data = (
            obs.astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            next_obs.astype(np.float32),
            dones.astype(np.float32),
            state.astype(np.float32),
            next_state.astype(np.float32),
        )
        # New transitions get max priority so they're sampled at least once
        self.tree.add(self.max_priority ** self.alpha, data)

    def sample(self, batch_size: int, device: torch.device, timestep: int = 0):
        # Anneal beta toward 1.0 over training
        progress = min(timestep / self.total_steps, 1.0)
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * progress

        indices = []
        priorities = []
        batch = []

        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            for _ in range(10):  # retry if None
                lo, hi = segment * i, segment * (i + 1)
                s = np.random.uniform(lo, hi)
                idx, priority, data = self.tree.get(s)
                if data is not None:
                    break
                s = np.random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            if data is None:
                continue
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        # Pad if needed
        while len(batch) < batch_size:
            s = np.random.uniform(0, self.tree.total())
            idx, priority, data = self.tree.get(s)
            if data is not None:
                indices.append(idx)
                priorities.append(priority)
                batch.append(data)

        # Importance sampling weights
        total = self.tree.total()
        n = len(self.tree)
        probs = np.array(priorities) / (total + 1e-8)
        weights = (n * probs + 1e-8) ** (-self.beta)
        weights /= weights.max()
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

        obs, actions, rewards, next_obs, dones, states, next_states = zip(*batch)
        return {
            "obs":        torch.tensor(np.stack(obs),         device=device),
            "actions":    torch.tensor(np.stack(actions),     device=device),
            "rewards":    torch.tensor(np.stack(rewards),     device=device),
            "next_obs":   torch.tensor(np.stack(next_obs),    device=device),
            "dones":      torch.tensor(np.stack(dones),       device=device),
            "states":     torch.tensor(np.stack(states),      device=device),
            "next_states":torch.tensor(np.stack(next_states), device=device),
            "weights":    weights_t,
            "indices":    indices,
        }

    def update_priorities(self, indices: list, td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            priority = (abs(err) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)
