"""
Neural networks for MAPPO.

Actor  — local observation → action distribution  (decentralized execution)
Critic — global state      → scalar value         (centralized training)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Policy network shared across all agents (parameter sharing)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dim, action_dim, n_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))

    def act(self, obs: torch.Tensor):
        dist = self.get_distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.get_distribution(obs)
        return dist.log_prob(actions), dist.entropy()


class Critic(nn.Module):
    """Centralized value network — takes global state, outputs scalar value."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dim, 1, n_layers)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)
