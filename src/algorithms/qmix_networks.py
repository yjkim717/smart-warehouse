"""
Neural networks for QMIX.

AgentQNetwork  — local obs → Q(s, a) for each action  (decentralized execution)
HyperNetwork   — global state → mixing weights         (centralized training)
MixingNetwork  — agent Q-values + state → Q_tot        (centralized training)

Reference: Rashid et al., "QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent Reinforcement Learning", ICML 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """Build MLP with orthogonal initialization."""
    gain = nn.init.calculate_gain("relu")
    layers = []
    d = in_dim
    for _ in range(n_layers):
        linear = nn.Linear(d, hidden_dim)
        nn.init.orthogonal_(linear.weight, gain=gain)
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.ReLU()]
        d = hidden_dim
    output = nn.Linear(d, out_dim)
    nn.init.orthogonal_(output.weight, gain=1.0)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class AgentQNetwork(nn.Module):
    """
    Per-agent Q-network: local obs → Q-values for each action.
    Shared weights across all agents (parameter sharing).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dim, action_dim, n_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (..., obs_dim) → Q: (..., action_dim)"""
        return self.net(obs)


class MixingNetwork(nn.Module):
    """
    QMIX Mixing Network.

    Takes individual agent Q-values and global state, produces a single
    monotonic Q_tot via hypernetworks. Monotonicity is enforced by passing
    hypernetwork outputs through abs() so mixing weights are non-negative.

    Architecture:
        hyper_w1: state → |weights| for first mixing layer  (n_agents → mixing_dim)
        hyper_b1: state → bias for first mixing layer
        hyper_w2: state → |weights| for output layer        (mixing_dim → 1)
        hyper_b2: state → scalar bias (via 2-layer MLP, no abs needed)
    """

    def __init__(self, n_agents: int, state_dim: int, mixing_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_dim = mixing_dim

        # Hypernetworks — generate mixing weights from global state
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, n_agents * mixing_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, mixing_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_qs : (batch, n_agents) — chosen Q-values per agent
            state    : (batch, state_dim) — global state

        Returns:
            q_tot : (batch,) — monotonic team Q-value
        """
        batch = agent_qs.size(0)

        # First mixing layer
        w1 = torch.abs(self.hyper_w1(state))                    # (batch, n_agents * mixing_dim)
        w1 = w1.view(batch, self.n_agents, self.mixing_dim)      # (batch, n_agents, mixing_dim)
        b1 = self.hyper_b1(state).unsqueeze(1)                   # (batch, 1, mixing_dim)

        qs = agent_qs.unsqueeze(1)                               # (batch, 1, n_agents)
        hidden = F.elu(torch.bmm(qs, w1) + b1)                  # (batch, 1, mixing_dim)

        # Output layer
        w2 = torch.abs(self.hyper_w2(state))                    # (batch, mixing_dim)
        w2 = w2.unsqueeze(2)                                     # (batch, mixing_dim, 1)
        b2 = self.hyper_b2(state)                                # (batch, 1)

        q_tot = torch.bmm(hidden, w2).squeeze(2) + b2           # (batch, 1)
        return q_tot.squeeze(1)                                  # (batch,)
