"""
MAPPO — Multi-Agent Proximal Policy Optimization (CTDE).

Centralized Training:  shared critic sees global state (all obs + agent ID)
Decentralized Execution: shared actor uses only local observation

Reference: Yu et al., "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games", NeurIPS 2021.
"""

import os
import numpy as np
import torch
import torch.nn as nn

from .networks import Actor, Critic
from .buffer import RolloutBuffer


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
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class MAPPO:

    def __init__(
        self,
        config: dict,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.device = torch.device(device)

        cfg = config["mappo"]
        self.gamma = cfg["gamma"]
        self.gae_lambda = cfg["gae_lambda"]
        self.clip_eps = cfg["clip_epsilon"]
        self.entropy_coef = cfg["entropy_coef"]
        self.value_loss_coef = cfg["value_loss_coef"]
        self.max_grad_norm = cfg["max_grad_norm"]
        self.n_epochs = cfg["n_epochs"]
        self.n_steps = cfg["n_steps"]
        self.minibatch_size = cfg["minibatch_size"]

        self.hidden_dim = cfg["hidden_dim"]
        self.n_layers = cfg["n_layers"]

        # Critic input = all agents' obs concatenated + one-hot agent ID
        self.global_obs_dim = n_agents * obs_dim + n_agents

        self.actor = Actor(obs_dim, action_dim, self.hidden_dim, self.n_layers).to(self.device)
        self.critic = Critic(self.global_obs_dim, self.hidden_dim, self.n_layers).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr_actor"])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr_critic"])

        self.buffer = RolloutBuffer(self.n_steps, n_agents, obs_dim, self.global_obs_dim)

        # Observation normalization (stabilizes learning)
        self.obs_rms = RunningMeanStd((obs_dim,))

    # ------------------------------------------------------------------
    # Global state construction
    # ------------------------------------------------------------------

    def build_global_obs(self, obs_list: list) -> np.ndarray:
        """
        Build per-agent global observations: concat(all local obs) + one-hot agent ID.
        Returns shape (n_agents, global_obs_dim).
        """
        all_obs = np.concatenate(obs_list)  # (n_agents * obs_dim,)
        global_obs = np.zeros((self.n_agents, self.global_obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            agent_id = np.zeros(self.n_agents, dtype=np.float32)
            agent_id[i] = 1.0
            global_obs[i] = np.concatenate([all_obs, agent_id])
        return global_obs

    # ------------------------------------------------------------------
    # Action selection (used during rollout collection)
    # ------------------------------------------------------------------

    def _normalize_obs(self, obs_array: np.ndarray, update: bool = True) -> np.ndarray:
        """Normalize observations using running statistics."""
        if update:
            self.obs_rms.update(obs_array)
        return self.obs_rms.normalize(obs_array).astype(np.float32)

    @torch.no_grad()
    def select_actions(self, obs_list: list):
        raw_obs = np.stack(obs_list)
        norm_obs = self._normalize_obs(raw_obs)
        obs_t = torch.tensor(norm_obs, device=self.device)
        actions, log_probs = self.actor.act(obs_t)

        global_obs = self.build_global_obs(
            [norm_obs[i] for i in range(self.n_agents)]
        )
        global_obs_t = torch.tensor(global_obs, device=self.device)
        values = self.critic(global_obs_t)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            global_obs,
            norm_obs,  # store normalized obs in buffer
        )

    @torch.no_grad()
    def get_values(self, obs_list: list) -> np.ndarray:
        raw_obs = np.stack(obs_list)
        norm_obs = self._normalize_obs(raw_obs, update=False)
        global_obs = self.build_global_obs(
            [norm_obs[i] for i in range(self.n_agents)]
        )
        global_obs_t = torch.tensor(global_obs, device=self.device)
        return self.critic(global_obs_t).cpu().numpy()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self):
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.minibatch_size, self.device):
                log_probs, entropy = self.actor.evaluate(batch["obs"], batch["actions"])
                values = self.critic(batch["global_obs"])

                adv = batch["advantages"]

                # Clipped surrogate objective
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch["returns"])
                entropy_mean = entropy.mean()

                # Actor step
                actor_loss = policy_loss - self.entropy_coef * entropy_mean
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                # Critic step
                critic_loss = self.value_loss_coef * value_loss
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_mean.item()
                n_updates += 1

        self.buffer.reset()
        return {k: v / max(n_updates, 1) for k, v in stats.items()}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "obs_rms": {"mean": self.obs_rms.mean, "var": self.obs_rms.var, "count": self.obs_rms.count},
                "metadata": {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "n_agents": self.n_agents,
                    "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers,
                },
            },
            path,
        )
        print(f"[mappo] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        if "obs_rms" in ckpt:
            self.obs_rms.mean = ckpt["obs_rms"]["mean"]
            self.obs_rms.var = ckpt["obs_rms"]["var"]
            self.obs_rms.count = ckpt["obs_rms"]["count"]
        print(f"[mappo] Loaded checkpoint ← {path}")
