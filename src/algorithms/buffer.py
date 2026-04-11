"""
Rollout buffer for on-policy MAPPO training.

Stores one rollout of (n_steps × n_agents) transitions, computes GAE
advantages, and yields shuffled minibatches for PPO updates.

When use_gru=True, also stores the GRU hidden state captured before each
step so the PPO update can reconstruct the correct temporal context for
every mini-batch item without needing to re-roll the full episode.
"""

import numpy as np
import torch


class RolloutBuffer:

    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_dim: int,
        global_obs_dim: int,
        use_gru: bool = False,
        hidden_dim: int = 0,
    ):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.use_gru = use_gru
        self.pos = 0

        self.obs = np.zeros((n_steps, n_agents, obs_dim), dtype=np.float32)
        self.global_obs = np.zeros((n_steps, n_agents, global_obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_agents), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.values = np.zeros((n_steps, n_agents), dtype=np.float32)

        self.returns = np.zeros_like(self.rewards)
        self.advantages = np.zeros_like(self.rewards)

        # GRU hidden states — shape (n_steps, n_agents, hidden_dim)
        # Stores the hidden state BEFORE each step so the update can use it.
        if use_gru and hidden_dim > 0:
            self.hidden_states = np.zeros((n_steps, n_agents, hidden_dim), dtype=np.float32)
        else:
            self.hidden_states = None

    # ------------------------------------------------------------------

    def insert(self, obs, global_obs, actions, log_probs, rewards, dones, values, hidden=None):
        self.obs[self.pos] = obs
        self.global_obs[self.pos] = global_obs
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.values[self.pos] = values
        if self.hidden_states is not None and hidden is not None:
            self.hidden_states[self.pos] = hidden  # (n_agents, hidden_dim)
        self.pos += 1

    def compute_returns(self, next_values: np.ndarray, gamma: float, gae_lambda: float):
        """Compute GAE-λ advantages and discounted returns."""
        gae = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            next_val = next_values if t == self.n_steps - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

        # Normalize advantages over the full rollout (not per-minibatch)
        adv_flat = self.advantages.reshape(-1)
        self.advantages = (self.advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    def get_batches(self, minibatch_size: int, device: torch.device):
        """Yield shuffled minibatches, flattening (steps, agents) → single batch dim."""
        total = self.n_steps * self.n_agents

        flat = {
            "obs": torch.tensor(self.obs.reshape(total, -1), device=device),
            "global_obs": torch.tensor(self.global_obs.reshape(total, -1), device=device),
            "actions": torch.tensor(self.actions.reshape(total), device=device),
            "old_log_probs": torch.tensor(self.log_probs.reshape(total), device=device),
            "returns": torch.tensor(self.returns.reshape(total), device=device),
            "advantages": torch.tensor(self.advantages.reshape(total), device=device),
            "old_values": torch.tensor(self.values.reshape(total), device=device),
        }

        if self.hidden_states is not None:
            # hidden_states: (n_steps, n_agents, hidden_dim) → (total, 1, hidden_dim)
            # The GRU expects (n_layers, batch, hidden_dim); we keep n_layers=1 dim here
            h = self.hidden_states.reshape(total, -1)
            flat["hidden_states"] = torch.tensor(h, device=device)

        indices = np.random.permutation(total)
        for start in range(0, total, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield {k: v[idx] for k, v in flat.items()}

    def reset(self):
        self.pos = 0
