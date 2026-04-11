"""
train_qmix_per.py — QMIX with Prioritized Experience Replay + Reward Shaping.

Improvements over train_qmix.py:
  1. PER: high-TD-error transitions sampled more often → success experiences
     not diluted by random failures in the buffer
  2. Reward Shaping: +pickup_reward when agent picks up a requested shelf,
     so the agent gets dense learning signal before the rare delivery reward

Usage:
    python scripts/train_qmix_per.py
    python scripts/train_qmix_per.py --env-config configs/env_config.yaml \
                                      --qmix-config configs/qmix_config_per.yaml
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from src.env.warehouse_env import WarehouseEnv
from src.algorithms.qmix import QMIX, RunningMeanStd
from src.algorithms.per_replay_buffer import PrioritizedReplayBuffer
from src.analytics import RewardTracker


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Reward shaping helper
# ------------------------------------------------------------------

def compute_shaped_rewards(
    env: WarehouseEnv,
    prev_carrying: list,
    rews: list,
    pickup_bonus: float,
    dropoff_bonus: float,
) -> np.ndarray:
    """
    Add dense reward shaping on top of raw rware rewards.

    Bonuses:
      +pickup_bonus  — agent just picked up a shelf that's in request_queue
      +dropoff_bonus — agent just put down a shelf (returned after delivery)
    """
    u = env._env.unwrapped
    requested_ids = {id(s) for s in u.request_queue}
    shaped = np.array(rews, dtype=np.float32)

    for i, agent in enumerate(u.agents):
        was_carrying = prev_carrying[i]
        now_carrying = agent.carrying_shelf

        # Just picked up a requested shelf
        if now_carrying is not None and was_carrying is None:
            if id(now_carrying) in requested_ids:
                shaped[i] += pickup_bonus

        # Just put down a shelf (returned after delivery)
        if now_carrying is None and was_carrying is not None:
            shaped[i] += dropoff_bonus

    return shaped


def get_carrying_state(env: WarehouseEnv) -> list:
    """Return list of carrying_shelf per agent (None or shelf object)."""
    u = env._env.unwrapped
    return [agent.carrying_shelf for agent in u.agents]


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(env: WarehouseEnv, qmix: QMIX, n_episodes: int, max_steps: int) -> dict:
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        for step in range(max_steps):
            actions = qmix.select_actions(obs, explore=False)
            obs, rews, dones, _ = env.step(actions.tolist())
            total_reward += sum(rews)
            if all(dones):
                break
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

    return {
        "mean_reward":   np.mean(episode_rewards),
        "std_reward":    np.std(episode_rewards),
        "max_reward":    np.max(episode_rewards),
        "mean_length":   np.mean(episode_lengths),
        "positive_rate": float(np.mean(np.array(episode_rewards) > 0)),
    }


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def plot_training_curve(train_log: list, tracker, log_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json

    plot_dir = "results/plots"
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if train_log:
        ts    = [e["timestep"] for e in train_log]
        means = [e["eval_mean_reward"] for e in train_log]
        stds  = [e["eval_std_reward"]  for e in train_log]

        means_arr = np.array(means)
        stds_arr  = np.array(stds)

        axes[0].plot(ts, means, color="#1976D2", linewidth=1.5, label="QMIX+PER eval mean")
        axes[0].fill_between(ts, means_arr - stds_arr, means_arr + stds_arr,
                             alpha=0.2, color="#1976D2")

        for path, label, color in [
            ("results/logs/random_baseline_rewards.json", "Random (0.066)", "#EF5350"),
            ("results/logs/greedy_baseline_rewards.json", "Greedy (0.443)", "#66BB6A"),
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                val = data["summary"]["team_total_reward"]["mean"]
                axes[0].axhline(val, color=color, linestyle="--", linewidth=1, label=label)

        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Team Total Reward")
        axes[0].set_title("QMIX+PER Eval Reward over Training")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    if tracker.episodes:
        ep_rewards = [ep["team_total_reward"] for ep in tracker.episodes]
        axes[1].plot(ep_rewards, color="#90CAF9", alpha=0.3, linewidth=0.5)
        window = min(100, max(1, len(ep_rewards) // 10))
        if len(ep_rewards) >= window:
            rolling = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
            axes[1].plot(range(window - 1, len(ep_rewards)), rolling,
                         color="#1976D2", linewidth=1.5, label=f"Rolling mean ({window})")
        axes[1].set_xlabel("Training Episode")
        axes[1].set_ylabel("Team Total Reward")
        axes[1].set_title("Training Episode Rewards (shaped)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    if train_log:
        ts = [e["timestep"] for e in train_log]
        axes[2].plot(ts, [e["loss"] for e in train_log], color="#EF5350", label="TD loss")
        ax2 = axes[2].twinx()
        ax2.plot(ts, [e["epsilon"] for e in train_log],
                 color="#AB47BC", linestyle="--", label="Epsilon")
        ax2.set_ylabel("Epsilon", color="#AB47BC")
        axes[2].set_xlabel("Timestep")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("TD Loss & Epsilon")
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2)
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "qmix_per_training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Training plot saved → {path}")


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(env_config: dict, qmix_config: dict, resume_path: str = None):
    cfg       = qmix_config["qmix"]
    total_ts  = int(cfg["total_timesteps"])
    eval_int  = cfg["eval_interval"]
    save_int  = cfg["save_interval"]
    log_int   = cfg["log_interval"]
    eval_eps  = cfg["eval_episodes"]
    ckpt_dir  = cfg["checkpoint_dir"]
    log_dir   = cfg["log_dir"]
    max_steps = env_config["env"].get("max_steps", 500)

    pickup_bonus  = cfg.get("pickup_reward", 0.3)
    dropoff_bonus = cfg.get("dropoff_reward", 0.1)
    per_alpha     = cfg.get("per_alpha", 0.6)
    per_beta_start = cfg.get("per_beta_start", 0.4)
    per_beta_end   = cfg.get("per_beta_end", 1.0)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    env      = WarehouseEnv(env_config)
    eval_env = WarehouseEnv(env_config)

    qmix = QMIX(qmix_config, env.obs_dim, env.action_dim, env.n_agents,
                device=device, total_timesteps=total_ts)
    if resume_path:
        qmix.load(resume_path)

    # Replace uniform buffer with PER buffer
    state_dim = env.n_agents * env.obs_dim
    per_buffer = PrioritizedReplayBuffer(
        capacity=cfg["buffer_size"],
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        state_dim=state_dim,
        alpha=per_alpha,
        beta_start=per_beta_start,
        beta_end=per_beta_end,
        total_steps=total_ts,
    )

    tracker = RewardTracker(n_agents=env.n_agents)

    print("=" * 60)
    print("QMIX + PER + Reward Shaping Training")
    print("=" * 60)
    print(f"  Env          : {env_config['env']['name']}")
    print(f"  Agents       : {env.n_agents}")
    print(f"  Obs dim      : {env.obs_dim}")
    print(f"  Device       : {device}")
    print(f"  Total ts     : {total_ts:,}")
    print(f"  Pickup bonus : +{pickup_bonus}")
    print(f"  PER alpha    : {per_alpha}")
    print("=" * 60)

    obs = env.reset()
    prev_carrying = get_carrying_state(env)
    ep_rewards  = np.zeros(env.n_agents, dtype=np.float64)
    ep_steps    = 0
    ep_count    = 0
    best_reward = -float("inf")
    recent_ep_rewards = []
    train_log   = []
    last_loss   = 0.0
    t_start     = time.time()
    timestep    = 0

    while timestep < total_ts:
        actions = qmix.select_actions(obs, explore=True)
        next_obs, rews, dones, _ = env.step(actions.tolist())

        raw_obs      = np.stack(obs)
        raw_next_obs = np.stack(next_obs)
        norm_obs     = qmix._normalize_obs(raw_obs)
        norm_next    = qmix._normalize_obs(raw_next_obs, update=False)
        state      = qmix.build_state([norm_obs[i]  for i in range(env.n_agents)])
        next_state = qmix.build_state([norm_next[i] for i in range(env.n_agents)])

        # Push rewards to PER buffer (reward shaping already applied by WarehouseEnv)
        per_buffer.push(norm_obs, actions, np.array(rews, dtype=np.float32), norm_next,
                        np.array(dones, dtype=np.float32), state, next_state)

        ep_rewards += rews  # track original rewards for logging
        ep_steps   += 1
        timestep   += 1
        obs         = next_obs

        # ---- Train with PER ----
        if len(per_buffer) >= cfg["train_start"]:
            batch = per_buffer.sample(cfg["batch_size"],
                                      torch.device(device),
                                      timestep=timestep)
            weights = batch["weights"]
            indices = batch["indices"]

            # Compute TD errors for priority update
            with torch.no_grad():
                B = batch["obs"].size(0)
                obs_flat  = batch["obs"].view(B * env.n_agents, env.obs_dim)
                nobs_flat = batch["next_obs"].view(B * env.n_agents, env.obs_dim)

                q_online  = qmix.agent_net(obs_flat).view(B, env.n_agents, env.action_dim)
                chosen_q  = q_online.gather(2, batch["actions"].unsqueeze(2)).squeeze(2)
                q_tot_cur = qmix.mixer(chosen_q, batch["states"])

                nq_online = qmix.agent_net(nobs_flat).view(B, env.n_agents, env.action_dim)
                best_a    = nq_online.argmax(dim=2, keepdim=True)
                nq_target = qmix.target_net(nobs_flat).view(B, env.n_agents, env.action_dim)
                nq_chosen = nq_target.gather(2, best_a).squeeze(2)
                nq_tot    = qmix.target_mixer(nq_chosen, batch["next_states"])

                team_r    = batch["rewards"].sum(dim=1)
                team_done = batch["dones"].max(dim=1).values
                target    = (team_r + qmix.gamma * nq_tot * (1 - team_done)).clamp(-50, 50)
                td_errors  = (q_tot_cur - target).abs().cpu().numpy()

            per_buffer.update_priorities(indices, td_errors)

            # Weighted loss
            obs_flat  = batch["obs"].view(B * env.n_agents, env.obs_dim)
            q_vals    = qmix.agent_net(obs_flat).view(B, env.n_agents, env.action_dim)
            chosen_q  = q_vals.gather(2, batch["actions"].unsqueeze(2)).squeeze(2)
            q_tot     = qmix.mixer(chosen_q, batch["states"])

            with torch.no_grad():
                nobs_flat = batch["next_obs"].view(B * env.n_agents, env.obs_dim)
                nq_online = qmix.agent_net(nobs_flat).view(B, env.n_agents, env.action_dim)
                best_a    = nq_online.argmax(dim=2, keepdim=True)
                nq_target = qmix.target_net(nobs_flat).view(B, env.n_agents, env.action_dim)
                nq_chosen = nq_target.gather(2, best_a).squeeze(2)
                nq_tot    = qmix.target_mixer(nq_chosen, batch["next_states"])
                team_r    = batch["rewards"].sum(dim=1)
                team_done = batch["dones"].max(dim=1).values
                target    = (team_r + qmix.gamma * nq_tot * (1 - team_done)).clamp(-50, 50)

            td_sq  = (q_tot - target) ** 2
            loss   = (weights * td_sq).mean()

            qmix.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qmix.params, qmix.max_grad_norm)
            qmix.optimizer.step()
            qmix._update_count += 1
            if qmix._update_count % qmix.target_update_interval == 0:
                qmix.target_net.load_state_dict(qmix.agent_net.state_dict())
                qmix.target_mixer.load_state_dict(qmix.mixer.state_dict())
            last_loss = loss.item()

        qmix.step_schedulers(timestep)

        # ---- Episode boundary ----
        if all(dones):
            team_total = float(ep_rewards.sum())
            tracker.episodes.append({
                "episode": ep_count,
                "n_steps": ep_steps,
                "duration_s": 0.0,
                "agent_total_rewards": ep_rewards.tolist(),
                "team_total_reward": round(team_total, 4),
                "team_reward_per_step": round(team_total / ep_steps, 6) if ep_steps else 0.0,
                "step_rewards": [],
            })
            recent_ep_rewards.append(team_total)
            ep_count  += 1
            ep_rewards = np.zeros(env.n_agents, dtype=np.float64)
            ep_steps   = 0
            obs        = env.reset()
            prev_carrying = get_carrying_state(env)

        # ---- Logging ----
        if timestep % log_int < 1 and recent_ep_rewards:
            elapsed = time.time() - t_start
            fps     = timestep / elapsed if elapsed > 0 else 0
            mean_r  = np.mean(recent_ep_rewards[-50:])
            print(
                f"  t={timestep:>9,}  ep={ep_count:<6}  "
                f"mean_r(50)={mean_r:>7.3f}  "
                f"loss={last_loss:>8.4f}  "
                f"eps={qmix.epsilon:.3f}  "
                f"lr={qmix.get_lr():.2e}  "
                f"fps={fps:>5.0f}"
            )

        # ---- Evaluation ----
        if timestep % eval_int < 1:
            eval_stats = evaluate(eval_env, qmix, eval_eps, max_steps)
            print(
                f"  [eval] t={timestep:>9,}  "
                f"reward={eval_stats['mean_reward']:.3f} ± {eval_stats['std_reward']:.3f}  "
                f"max={eval_stats['max_reward']:.1f}  "
                f"pos_rate={eval_stats['positive_rate']:.1%}"
            )
            train_log.append({
                "timestep":         timestep,
                "eval_mean_reward": round(eval_stats["mean_reward"], 4),
                "eval_std_reward":  round(eval_stats["std_reward"],  4),
                "eval_max_reward":  round(eval_stats["max_reward"],  4),
                "positive_rate":    round(eval_stats["positive_rate"], 4),
                "loss":             round(last_loss, 6),
                "epsilon":          round(qmix.epsilon, 4),
                "lr":               round(qmix.get_lr(), 8),
            })
            if eval_stats["mean_reward"] > best_reward:
                best_reward = eval_stats["mean_reward"]
                os.makedirs(ckpt_dir, exist_ok=True)
                qmix.save(os.path.join(ckpt_dir, "qmix_per_best.pt"))

        # ---- Periodic save ----
        if timestep % save_int < 1:
            os.makedirs(ckpt_dir, exist_ok=True)
            qmix.save(os.path.join(ckpt_dir, "qmix_per_latest.pt"))

    # ---- Final ----
    qmix.save(os.path.join(ckpt_dir, "qmix_per_latest.pt"))
    elapsed = time.time() - t_start
    print(f"\nTraining complete — {timestep:,} timesteps in {elapsed:.1f}s")
    print(f"Best eval reward: {best_reward:.4f}")

    os.makedirs(log_dir, exist_ok=True)
    tracker.save(os.path.join(log_dir, "qmix_per_train_rewards.json"), include_step_trace=False)
    tracker.save_csv(os.path.join(log_dir, "qmix_per_train_rewards.csv"))

    import json
    with open(os.path.join(log_dir, "qmix_per_eval_curve.json"), "w") as f:
        json.dump(train_log, f, indent=2)

    plot_training_curve(train_log, tracker, log_dir)
    env.close()
    eval_env.close()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config",  default="configs/env_config.yaml")
    parser.add_argument("--qmix-config", default="configs/qmix_config_per.yaml")
    parser.add_argument("--resume",      default=None)
    args = parser.parse_args()

    env_config  = load_config(args.env_config)
    qmix_config = load_config(args.qmix_config)
    train(env_config, qmix_config, resume_path=args.resume)


if __name__ == "__main__":
    main()
