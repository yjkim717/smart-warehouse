"""
Neural networks for MAPPO.

Actor    — local observation → action distribution  (decentralized execution)
GRUActor — recurrent variant with temporal context  (decentralized execution)
Critic   — global state      → scalar value         (centralized training)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _build_mlp(
    in_dim: int, hidden_dim: int, out_dim: int, n_layers: int,
    output_gain: float = 1.0,
) -> nn.Sequential:
    """Build MLP with orthogonal initialization (MAPPO paper, Yu et al. 2021)."""
    gain = nn.init.calculate_gain("tanh")
    layers = []
    d = in_dim
    for _ in range(n_layers):
        linear = nn.Linear(d, hidden_dim)
        nn.init.orthogonal_(linear.weight, gain=gain)
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
        d = hidden_dim
    output = nn.Linear(d, out_dim)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Policy network shared across all agents (parameter sharing)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dim, action_dim, n_layers, output_gain=0.01)

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


class GRUActor(nn.Module):
    """
    Recurrent policy network — maintains temporal context across steps.

    Architecture: Linear encoder → GRU → Linear output
    Hidden state is carried step-by-step during rollout collection and
    reset at episode boundaries. During the PPO update, stored per-step
    hidden states are used so each mini-batch item has the correct context.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        gain = nn.init.calculate_gain("tanh")
        enc_linear = nn.Linear(obs_dim, hidden_dim)
        nn.init.orthogonal_(enc_linear.weight, gain=gain)
        nn.init.zeros_(enc_linear.bias)
        self.encoder = nn.Sequential(enc_linear, nn.Tanh())

        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layers, batch_first=False)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        self.output = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """
        Step-by-step forward pass (used during rollout and single-step evaluation).

        obs    : (batch, obs_dim)
        hidden : (n_layers, batch, hidden_dim)
        Returns: logits (batch, action_dim), new_hidden (n_layers, batch, hidden_dim)
        """
        x = self.encoder(obs).unsqueeze(0)          # (1, batch, hidden_dim)
        gru_out, new_hidden = self.gru(x, hidden)   # (1, batch, hidden_dim)
        logits = self.output(gru_out.squeeze(0))    # (batch, action_dim)
        return logits, new_hidden

    def forward_sequence(self, obs_seq: torch.Tensor, hidden: torch.Tensor):
        """
        Sequence forward pass (used during PPO update with stored hidden states).

        obs_seq : (seq_len, batch, obs_dim)
        hidden  : (n_layers, batch, hidden_dim)
        Returns : logits (seq_len * batch, action_dim), new_hidden
        """
        seq_len, batch, _ = obs_seq.shape
        x = self.encoder(obs_seq.reshape(seq_len * batch, -1))
        x = x.reshape(seq_len, batch, -1)           # (seq, batch, hidden_dim)
        gru_out, new_hidden = self.gru(x, hidden)   # (seq, batch, hidden_dim)
        logits = self.output(gru_out.reshape(seq_len * batch, -1))
        return logits, new_hidden

    def get_distribution(self, obs: torch.Tensor, hidden: torch.Tensor):
        logits, new_hidden = self.forward(obs, hidden)
        return Categorical(logits=logits), new_hidden

    def act(self, obs: torch.Tensor, hidden: torch.Tensor):
        dist, new_hidden = self.get_distribution(obs, hidden)
        action = dist.sample()
        return action, dist.log_prob(action), new_hidden

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor, hidden: torch.Tensor):
        """Evaluate log-probs and entropy using stored per-step hidden states."""
        logits, _ = self.forward(obs, hidden)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zeroed initial hidden state."""
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)


class Critic(nn.Module):
    """Centralized value network — takes global state, outputs scalar value."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dim, 1, n_layers, output_gain=1.0)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)
