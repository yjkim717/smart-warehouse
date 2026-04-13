"""
HAPPO — Heterogeneous-Agent Proximal Policy Optimization (CTDE).

Algorithmic differences vs MAPPO:
  - Each agent has its OWN independent actor network (no parameter sharing)
  - Sequential update scheme: agents are updated one at a time in a
    randomized order each epoch
  - M-factor: when updating agent i, the GAE advantage is reweighted by the
    product of IS ratios of all previously-updated agents in this epoch:
      happo_adv_i[t] = gae_adv[t, i] * M_weights[t]
      M_weights[t] *= π_j_new(a_j_t | o_j_t) / π_j_old(a_j_t | o_j_t)
    This ensures a monotonic improvement guarantee in cooperative MARL.
  - Shared centralized critic (same as MAPPO)

Reuses from MAPPO:
  - RunningMeanStd (obs normalization)
  - Actor, Critic (networks.py)
  - RolloutBuffer (buffer.py, accessed via raw array slices)
  - All scheduler math (cosine LR decay, entropy annealing, clip decay)

Reference: Kuba et al., "Trust Region Policy Optimisation in Multi-Agent
Reinforcement Learning", ICLR 2022.
"""

import os
import numpy as np
import torch
import torch.nn as nn

from .networks import Actor, Critic
from .buffer import RolloutBuffer
from .mappo import RunningMeanStd   # reuse — no redefinition


class HAPPO:

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

        cfg = config["happo"]

        # ---- Core PPO hyperparameters ----
        self.gamma = cfg["gamma"]
        self.gae_lambda = cfg["gae_lambda"]
        self.value_loss_coef = cfg["value_loss_coef"]
        self.max_grad_norm = cfg["max_grad_norm"]
        self.n_epochs = cfg["n_epochs"]
        self.n_steps = cfg["n_steps"]
        self.minibatch_size = cfg["minibatch_size"]

        self.hidden_dim = cfg["hidden_dim"]
        self.n_layers = cfg["n_layers"]

        # ---- Clip epsilon decay ----
        if "clip_epsilon_start" in cfg:
            self.clip_eps_start = cfg["clip_epsilon_start"]
            self.clip_eps_end = cfg["clip_epsilon_end"]
        else:
            self.clip_eps_start = cfg["clip_epsilon"]
            self.clip_eps_end = cfg["clip_epsilon"]
        self.clip_eps = self.clip_eps_start

        # ---- Entropy annealing ----
        if "entropy_coef_start" in cfg:
            self.entropy_coef_start = cfg["entropy_coef_start"]
            self.entropy_coef_end = cfg["entropy_coef_end"]
        else:
            self.entropy_coef_start = cfg["entropy_coef"]
            self.entropy_coef_end = cfg["entropy_coef"]
        self.entropy_coef = self.entropy_coef_start

        # ---- LR decay ----
        self.lr_actor_base = cfg["lr_actor"]
        self.lr_critic_base = cfg["lr_critic"]
        self.lr_decay = cfg.get("lr_decay", False)
        self.lr_min = cfg.get("lr_min", 1e-5)

        # ---- Training schedule ----
        self.total_timesteps = total_timesteps or cfg.get("total_timesteps", 2_000_000)
        self._current_timestep = 0

        # ---- Networks ----
        # Critic input: all agents' obs concatenated + one-hot agent ID
        self.global_obs_dim = n_agents * obs_dim + n_agents

        # Each agent gets its own independent actor — no parameter sharing
        self.actors = nn.ModuleList([
            Actor(obs_dim, action_dim, self.hidden_dim, self.n_layers).to(self.device)
            for _ in range(n_agents)
        ])

        self.critic = Critic(self.global_obs_dim, self.hidden_dim, self.n_layers).to(self.device)

        # One optimizer per actor (independent updates)
        self.actor_optims = [
            torch.optim.Adam(self.actors[i].parameters(), lr=self.lr_actor_base)
            for i in range(n_agents)
        ]
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic_base)

        self.buffer = RolloutBuffer(
            self.n_steps,
            n_agents,
            obs_dim,
            self.global_obs_dim,
        )

        # One shared obs normalizer (homogeneous agents — same obs space)
        self.obs_rms = RunningMeanStd((obs_dim,))

    # ------------------------------------------------------------------
    # Scheduler helpers — identical math to MAPPO, loops over actor_optims
    # ------------------------------------------------------------------

    def step_schedulers(self, timestep: int):
        """Update all scheduled hyperparameters based on current timestep."""
        self._current_timestep = timestep
        progress = min(timestep / self.total_timesteps, 1.0)

        # Cosine LR decay
        if self.lr_decay:
            cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            new_lr_actor = self.lr_min + (self.lr_actor_base - self.lr_min) * cosine_factor
            new_lr_critic = self.lr_min + (self.lr_critic_base - self.lr_min) * cosine_factor
            for optim in self.actor_optims:
                for pg in optim.param_groups:
                    pg["lr"] = new_lr_actor
            for pg in self.critic_optim.param_groups:
                pg["lr"] = new_lr_critic

        # Linear entropy annealing
        self.entropy_coef = max(
            self.entropy_coef_end,
            self.entropy_coef_start - (self.entropy_coef_start - self.entropy_coef_end) * progress,
        )

        # Linear clip epsilon decay
        self.clip_eps = max(
            self.clip_eps_end,
            self.clip_eps_start - (self.clip_eps_start - self.clip_eps_end) * progress,
        )

    def get_lr(self) -> float:
        """Return current actor learning rate (from actor 0 by convention)."""
        return self.actor_optims[0].param_groups[0]["lr"]

    # ------------------------------------------------------------------
    # GRU stub — HAPPO uses MLP actors only
    # ------------------------------------------------------------------

    def reset_hidden(self):
        """No-op: HAPPO uses MLP actors, no recurrent state."""
        pass

    # ------------------------------------------------------------------
    # Global state construction (identical to MAPPO)
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
    # Observation normalization (identical to MAPPO)
    # ------------------------------------------------------------------

    def _normalize_obs(self, obs_array: np.ndarray, update: bool = True) -> np.ndarray:
        if update:
            self.obs_rms.update(obs_array)
        return self.obs_rms.normalize(obs_array).astype(np.float32)

    # ------------------------------------------------------------------
    # Action selection (used during rollout collection)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_actions(self, obs_list: list):
        """
        Sample actions for all agents using their independent actors.

        Return signature is identical to MAPPO.select_actions so the
        training loop works unchanged.

        Returns:
            actions     (n_agents,) int64
            log_probs   (n_agents,) float32
            values      (n_agents,) float32
            global_obs  (n_agents, global_obs_dim) float32
            norm_obs    (n_agents, obs_dim) float32
            hidden_np   None  (no GRU)
        """
        raw_obs = np.stack(obs_list)                    # (n_agents, obs_dim)
        norm_obs = self._normalize_obs(raw_obs)         # (n_agents, obs_dim)

        actions = np.zeros(self.n_agents, dtype=np.int64)
        log_probs = np.zeros(self.n_agents, dtype=np.float32)

        for i in range(self.n_agents):
            obs_i = torch.tensor(norm_obs[i:i+1], device=self.device)  # (1, obs_dim)
            a_i, lp_i = self.actors[i].act(obs_i)
            actions[i] = a_i.item()
            log_probs[i] = lp_i.item()

        global_obs = self.build_global_obs(
            [norm_obs[i] for i in range(self.n_agents)]
        )
        global_obs_t = torch.tensor(global_obs, device=self.device)
        values = self.critic(global_obs_t).cpu().numpy()

        return actions, log_probs, values, global_obs, norm_obs, None

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
    # HAPPO update — sequential actors with M-factor reweighting
    # ------------------------------------------------------------------

    def update(self):
        """
        HAPPO sequential PPO update.

        For each epoch:
          1. Randomly shuffle agent update order.
          2. Initialize M_weights = ones(n_steps) — IS product accumulator.
          3. For each agent in order:
               a. Compute HAPPO advantage = GAE_adv[:, agent_idx] * M_weights
               b. Run PPO minibatch update on actors[agent_idx] only
               c. Recompute IS ratios for all steps using updated actor
               d. M_weights *= IS_ratio  (no gradient)
          4. Update shared critic once after all actors.
        """
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "m_weight_mean": 0.0,   # diagnostic: average IS reweighting per epoch
        }
        n_actor_updates = 0
        n_critic_updates = 0

        for epoch in range(self.n_epochs):

            # ── Step 1: Sequential actor updates with M-factor ──────────────

            # Fresh random permutation of agents each epoch
            agent_order = np.random.permutation(self.n_agents)

            # Per-step IS product — reset each epoch, accumulated as agents update
            # Shape: (n_steps,) — indexed by rollout step, NOT by flattened sample
            M_weights = np.ones(self.n_steps, dtype=np.float32)

            for _agent_idx in agent_order:
                agent_idx = int(_agent_idx)

                # ── Slice per-agent data from buffer ──────────────────────────
                # All shapes: (n_steps, ...) — one rollout per agent
                obs_i = self.buffer.obs[:, agent_idx, :]         # (n_steps, obs_dim)
                actions_i = self.buffer.actions[:, agent_idx]    # (n_steps,)
                old_lp_i = self.buffer.log_probs[:, agent_idx]   # (n_steps,) from rollout
                returns_i = self.buffer.returns[:, agent_idx]    # (n_steps,)
                old_val_i = self.buffer.values[:, agent_idx]     # (n_steps,)
                # global_obs stored per-agent: (n_steps, n_agents, global_obs_dim)
                # → take agent_idx column: (n_steps, global_obs_dim)
                global_obs_i = self.buffer.global_obs[:, agent_idx, :]

                # ── HAPPO advantage: GAE * accumulated IS product ─────────────
                base_adv = self.buffer.advantages[:, agent_idx]  # (n_steps,)
                happo_adv = base_adv * M_weights                  # (n_steps,) elementwise

                # Normalize HAPPO advantage (over this agent's n_steps)
                happo_adv = (happo_adv - happo_adv.mean()) / (happo_adv.std() + 1e-8)

                # Convert to tensors
                obs_t = torch.tensor(obs_i, device=self.device)
                actions_t = torch.tensor(actions_i, device=self.device)
                old_lp_t = torch.tensor(old_lp_i, device=self.device)
                happo_adv_t = torch.tensor(happo_adv, device=self.device)

                # ── PPO minibatch update over n_steps for actors[agent_idx] ──
                indices = np.random.permutation(self.n_steps)

                for start in range(0, self.n_steps, self.minibatch_size):
                    idx = indices[start: start + self.minibatch_size]
                    idx_t = torch.tensor(idx, device=self.device, dtype=torch.long)

                    lp_new, ent = self.actors[agent_idx].evaluate(
                        obs_t[idx_t], actions_t[idx_t]
                    )

                    adv_mb = happo_adv_t[idx_t]

                    # Clipped surrogate (PPO-clip, same formula as MAPPO)
                    ratio = torch.exp(lp_new - old_lp_t[idx_t])
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()

                    actor_loss = policy_loss - self.entropy_coef * ent.mean()
                    self.actor_optims[agent_idx].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[agent_idx].parameters(), self.max_grad_norm
                    )
                    self.actor_optims[agent_idx].step()

                    stats["policy_loss"] += policy_loss.item()
                    stats["entropy"] += ent.mean().item()
                    n_actor_updates += 1

                # ── Update M_weights from newly-updated actors[agent_idx] ─────
                # Compute π_new(a_i | o_i) / π_old(a_i | o_i) for ALL n_steps
                # Important: use old_lp_i (rollout-time, unchanged) as denominator
                # M_weights must stay as NumPy — no gradient through them
                with torch.no_grad():
                    new_lp_all, _ = self.actors[agent_idx].evaluate(obs_t, actions_t)
                new_lp_np = new_lp_all.cpu().numpy()                    # (n_steps,)
                is_ratio = np.exp(new_lp_np - old_lp_i)                 # (n_steps,)
                M_weights = M_weights * is_ratio                         # accumulate product

            stats["m_weight_mean"] += float(M_weights.mean())

            # ── Step 2: Critic update (once per epoch, after all actors) ─────
            # Uses all (n_steps × n_agents) samples — same as MAPPO critic update
            total = self.n_steps * self.n_agents
            global_obs_flat = torch.tensor(
                self.buffer.global_obs.reshape(total, -1), device=self.device
            )
            returns_flat = torch.tensor(
                self.buffer.returns.reshape(total), device=self.device
            )
            old_values_flat = torch.tensor(
                self.buffer.values.reshape(total), device=self.device
            )

            c_indices = np.random.permutation(total)
            for start in range(0, total, self.minibatch_size):
                idx = torch.tensor(
                    c_indices[start: start + self.minibatch_size],
                    device=self.device, dtype=torch.long
                )

                values = self.critic(global_obs_flat[idx])
                old_v = old_values_flat[idx]
                ret = returns_flat[idx]

                # Clipped value loss (same as MAPPO)
                v_clipped = old_v + torch.clamp(values - old_v, -self.clip_eps, self.clip_eps)
                vf_unclip = (values - ret).pow(2)
                vf_clip = (v_clipped - ret).pow(2)
                value_loss = torch.max(vf_unclip, vf_clip).mean()

                critic_loss = self.value_loss_coef * value_loss
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                stats["value_loss"] += value_loss.item()
                n_critic_updates += 1

        self.buffer.reset()
        return {
            "policy_loss": stats["policy_loss"] / max(n_actor_updates, 1),
            "value_loss": stats["value_loss"] / max(n_critic_updates, 1),
            "entropy": stats["entropy"] / max(n_actor_updates, 1),
            "m_weight_mean": stats["m_weight_mean"] / max(self.n_epochs, 1),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actors": [actor.state_dict() for actor in self.actors],
                "critic": self.critic.state_dict(),
                "actor_optims": [opt.state_dict() for opt in self.actor_optims],
                "critic_optim": self.critic_optim.state_dict(),
                "obs_rms": {
                    "mean": self.obs_rms.mean,
                    "var": self.obs_rms.var,
                    "count": self.obs_rms.count,
                },
                "scheduler_state": {
                    "current_timestep": self._current_timestep,
                    "entropy_coef": self.entropy_coef,
                    "clip_eps": self.clip_eps,
                },
                "metadata": {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "n_agents": self.n_agents,
                    "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers,
                    "algo": "happo",
                },
            },
            path,
        )
        print(f"[happo] Saved checkpoint → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for i, state in enumerate(ckpt["actors"]):
            self.actors[i].load_state_dict(state)
        self.critic.load_state_dict(ckpt["critic"])
        for i, state in enumerate(ckpt["actor_optims"]):
            self.actor_optims[i].load_state_dict(state)
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        if "obs_rms" in ckpt:
            self.obs_rms.mean = ckpt["obs_rms"]["mean"]
            self.obs_rms.var = ckpt["obs_rms"]["var"]
            self.obs_rms.count = ckpt["obs_rms"]["count"]
        if "scheduler_state" in ckpt:
            ss = ckpt["scheduler_state"]
            self._current_timestep = ss.get("current_timestep", 0)
            self.entropy_coef = ss.get("entropy_coef", self.entropy_coef_start)
            self.clip_eps = ss.get("clip_eps", self.clip_eps_start)
        print(f"[happo] Loaded checkpoint ← {path}")
