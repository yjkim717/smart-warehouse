"""
Replay Buffer for off-policy QMIX training.

Stores (obs, actions, rewards, next_obs, dones, state, next_state) transitions
and samples random mini-batches for training.
"""

import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.buf = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,        # (n_agents, obs_dim)
        actions: np.ndarray,    # (n_agents,)
        rewards: np.ndarray,    # (n_agents,)
        next_obs: np.ndarray,   # (n_agents, obs_dim)
        dones: np.ndarray,      # (n_agents,)
        state: np.ndarray,      # (state_dim,)
        next_state: np.ndarray, # (state_dim,)
    ):
        self.buf.append((
            obs.astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            next_obs.astype(np.float32),
            dones.astype(np.float32),
            state.astype(np.float32),
            next_state.astype(np.float32),
        ))

    def sample(self, batch_size: int, device: torch.device) -> dict:
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones, states, next_states = zip(*batch)

        return {
            "obs":        torch.tensor(np.stack(obs),         device=device),   # (B, n_agents, obs_dim)
            "actions":    torch.tensor(np.stack(actions),     device=device),   # (B, n_agents)
            "rewards":    torch.tensor(np.stack(rewards),     device=device),   # (B, n_agents)
            "next_obs":   torch.tensor(np.stack(next_obs),    device=device),   # (B, n_agents, obs_dim)
            "dones":      torch.tensor(np.stack(dones),       device=device),   # (B, n_agents)
            "states":     torch.tensor(np.stack(states),      device=device),   # (B, state_dim)
            "next_states":torch.tensor(np.stack(next_states), device=device),   # (B, state_dim)
        }

    def __len__(self) -> int:
        return len(self.buf)
