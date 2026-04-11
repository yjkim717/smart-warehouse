"""
QMIX — Monotonic Value Function Factorisation for MARL.

Each agent has its own Q-network (decentralized execution).
A mixing network combines individual Q-values with global state
into a monotonic team Q-value (centralized training).

Key properties:
  - Off-policy learning with replay buffer
  - Epsilon-greedy exploration with linear decay
  - Target networks for stable training
  - Observation normalization (RunningMeanStd)

Reference: Rashid et al., "QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent Reinforcement Learning", ICML 2018.
"""

import os
import numpy as np
import torch
import torch.nn as nn

from .qmix_networks import AgentQNetwork, MixingNetwork
from .replay_buffer import ReplayBuffer


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance normalization."""

    def __init__(self, shape: tuple):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch: np.ndarray):
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class QMIX:

    def __init__(
        self,
        config: dict,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        device: str = "cpu",
        total_timesteps: int = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = torch.device(device)

        cfg = config["qmix"]

        # ---- Hyperparameters ----
        self.gamma        = cfg["gamma"]
        self.lr           = cfg["lr"]
        self.lr_min       = cfg.get("lr_min", 1e-5)
        self.lr_decay     = cfg.get("lr_decay", False)
        self.hidden_dim   = cfg["hidden_dim"]
        self.n_layers     = cfg["n_layers"]
        self.mixing_dim   = cfg["mixing_dim"]
        self.batch_size   = cfg["batch_size"]
        self.buffer_size  = cfg["buffer_size"]
        self.target_update_interval = cfg["target_update_interval"]
        self.max_grad_norm = cfg["max_grad_norm"]
        self.train_start   = cfg["train_start"]

        # ---- Epsilon decay ----
        self.epsilon       = cfg["epsilon_start"]
        self.epsilon_start = cfg["epsilon_start"]
        self.epsilon_end   = cfg["epsilon_end"]

        # ---- Training schedule ----
        self.total_timesteps = total_timesteps or cfg.get("total_timesteps", 2_000_000)
        self._current_timestep = 0

        # ---- Global state = concat of all agents' obs ----
        self.state_dim = n_agents * obs_dim

        # ---- Networks ----
        self.agent_net  = AgentQNetwork(obs_dim, action_dim, self.hidden_dim, self.n_layers).to(self.device)
        self.target_net = AgentQNetwork(obs_dim, action_dim, self.hidden_dim, self.n_layers).to(self.device)
        self.target_net.load_state_dict(self.agent_net.state_dict())
        self.target_net.eval()

        self.mixer        = MixingNetwork(n_agents, self.state_dim, self.mixing_dim).to(self.device)
        self.target_mixer = MixingNetwork(n_agents, self.state_dim, self.mixing_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.target_mixer.eval()

        # ---- Optimizer (agent net + mixer jointly) ----
        self.params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        # ---- Replay buffer ----
        self.buffer = ReplayBuffer(self.buffer_size, n_agents, obs_dim, self.state_dim)

        # ---- Observation normalization ----
        self.obs_rms = RunningMeanStd((obs_dim,))

        self._update_count = 0

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def build_state(self, obs_list: list) -> np.ndarray:
        """Concatenate all agents' observations into global state."""
        return np.concatenate(obs_list).astype(np.float32)  # (state_dim,)

    # ------------------------------------------------------------------
    # Observation normalization
    # ------------------------------------------------------------------

    def _normalize_obs(self, obs_array: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.obs_rms.update(obs_array)
        return self.obs_rms.normalize(obs_array).astype(np.float32)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_actions(self, obs_list: list, explore: bool = True) -> np.ndarray:
        """
        Epsilon-greedy action selection.

        Returns:
            actions: np.ndarray of shape (n_agents,)
        """
        raw_obs = np.stack(obs_list)
        norm_obs = self._normalize_obs(raw_obs, update=False)

        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim, size=self.n_agents)

        obs_t = torch.tensor(norm_obs, device=self.device)
        with torch.no_grad():
            q_values = self.agent_net(obs_t)  # (n_agents, action_dim)
        return q_values.argmax(dim=-1).cpu().numpy()

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def step_schedulers(self, timestep: int):
        """Update epsilon and optionally LR based on current timestep."""
        self._current_timestep = timestep
        progress = min(timestep / self.total_timesteps, 1.0)

        # Linear epsilon decay
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress,
        )

        # Cosine LR decay
        if self.lr_decay:
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            new_lr = self.lr_min + (self.lr - self.lr_min) * cosine_factor
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self) -> dict:
        if len(self.buffer) < self.train_start:
            return {}

        batch = self.buffer.sample(self.batch_size, self.device)

        obs        = batch["obs"]         # (B, n_agents, obs_dim)
        actions    = batch["actions"]     # (B, n_agents)
        rewards    = batch["rewards"]     # (B, n_agents)
        next_obs   = batch["next_obs"]    # (B, n_agents, obs_dim)
        dones      = batch["dones"]       # (B, n_agents)
        states     = batch["states"]      # (B, state_dim)
        next_states= batch["next_states"] # (B, state_dim)

        B = obs.size(0)

        # ---- Current Q-values ----
        # Flatten agents into batch dim for network forward pass
        obs_flat      = obs.view(B * self.n_agents, self.obs_dim)
        q_vals_flat   = self.agent_net(obs_flat)                       # (B*n, action_dim)
        q_vals        = q_vals_flat.view(B, self.n_agents, self.action_dim)
        chosen_q      = q_vals.gather(2, actions.unsqueeze(2)).squeeze(2)  # (B, n_agents)

        # ---- Target Q-values (Double DQN style) ----
        with torch.no_grad():
            next_obs_flat     = next_obs.view(B * self.n_agents, self.obs_dim)

            # Online net selects best action
            next_q_online     = self.agent_net(next_obs_flat).view(B, self.n_agents, self.action_dim)
            best_actions      = next_q_online.argmax(dim=2, keepdim=True)

            # Target net evaluates it
            next_q_target     = self.target_net(next_obs_flat).view(B, self.n_agents, self.action_dim)
            next_q_chosen     = next_q_target.gather(2, best_actions).squeeze(2)  # (B, n_agents)

        # ---- Mix Q-values ----
        q_tot         = self.mixer(chosen_q, states)                  # (B,)

        with torch.no_grad():
            next_q_tot = self.target_mixer(next_q_chosen, next_states) # (B,)

        # Team reward = sum of individual rewards
        team_rewards  = rewards.sum(dim=1)                            # (B,)
        # Done if ALL agents are done
        team_dones    = dones.max(dim=1).values                       # (B,)

        target = team_rewards + self.gamma * next_q_tot * (1.0 - team_dones)
        target = target.clamp(-50.0, 50.0)

        # ---- Loss ----
        loss = nn.MSELoss()(q_tot, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

        self._update_count += 1

        # ---- Sync target networks ----
        if self._update_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return {"loss": loss.item()}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "agent_net": self.agent_net.state_dict(),
                "mixer":     self.mixer.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_rms": {
                    "mean":  self.obs_rms.mean,
                    "var":   self.obs_rms.var,
                    "count": self.obs_rms.count,
                },
                "scheduler_state": {
                    "current_timestep": self._current_timestep,
                    "epsilon":          self.epsilon,
                },
                "metadata": {
                    "obs_dim":    self.obs_dim,
                    "action_dim": self.action_dim,
                    "n_agents":   self.n_agents,
                    "hidden_dim": self.hidden_dim,
                    "n_layers":   self.n_layers,
                    "mixing_dim": self.mixing_dim,
                },
            },
            path,
        )
        print(f"[qmix] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent_net.load_state_dict(ckpt["agent_net"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self.target_net.load_state_dict(ckpt["agent_net"])
        self.target_mixer.load_state_dict(ckpt["mixer"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "obs_rms" in ckpt:
            self.obs_rms.mean  = ckpt["obs_rms"]["mean"]
            self.obs_rms.var   = ckpt["obs_rms"]["var"]
            self.obs_rms.count = ckpt["obs_rms"]["count"]
        if "scheduler_state" in ckpt:
            ss = ckpt["scheduler_state"]
            self._current_timestep = ss.get("current_timestep", 0)
            self.epsilon           = ss.get("epsilon", self.epsilon_start)
        print(f"[qmix] Loaded checkpoint ← {path}")
