"""
train_qmix.py — Train QMIX agents in the warehouse environment.

Usage:
    python scripts/train_qmix.py
    python scripts/train_qmix.py --env-config configs/env_config.yaml --qmix-config configs/qmix_config.yaml
    python scripts/train_qmix.py --resume results/checkpoints/qmix_latest.pt
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np

sys.path.insert(0, ".")
from src.env.warehouse_env import WarehouseEnv
from src.algorithms.qmix import QMIX
from src.analytics import RewardTracker


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(env: WarehouseEnv, qmix: QMIX, n_episodes: int, max_steps: int) -> dict:
    """Run greedy evaluation (no exploration)."""
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

    plot_dir = "results/plots"
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if train_log:
        ts    = [e["timestep"] for e in train_log]
        means = [e["eval_mean_reward"] for e in train_log]
        stds  = [e["eval_std_reward"]  for e in train_log]

        means_arr = np.array(means)
        stds_arr  = np.array(stds)

        axes[0].plot(ts, means, color="#1976D2", linewidth=1.5, label="QMIX eval mean")
        axes[0].fill_between(ts, means_arr - stds_arr, means_arr + stds_arr,
                             alpha=0.2, color="#1976D2")

        random_path = os.path.join(log_dir, "random_baseline_rewards.json")
        greedy_path = os.path.join(log_dir, "greedy_baseline_rewards.json")
        import json
        for path, label, color in [
            (random_path, "Random (0.066)", "#EF5350"),
            (greedy_path, "Greedy (0.443)", "#66BB6A"),
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                val = data["summary"]["team_total_reward"]["mean"]
                axes[0].axhline(val, color=color, linestyle="--", linewidth=1, label=f"{label}")

        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Team Total Reward")
        axes[0].set_title("QMIX Eval Reward over Training")
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
        axes[1].set_title("Training Episode Rewards")
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
    path = os.path.join(plot_dir, "qmix_training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[train] Training plot saved → {path}")


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(env_config: dict, qmix_config: dict, resume_path: str = None):
    cfg        = qmix_config["qmix"]
    total_ts   = int(cfg["total_timesteps"])
    eval_int   = cfg["eval_interval"]
    save_int   = cfg["save_interval"]
    log_int    = cfg["log_interval"]
    eval_eps   = cfg["eval_episodes"]
    ckpt_dir   = cfg["checkpoint_dir"]
    log_dir    = cfg["log_dir"]
    max_steps  = env_config["env"].get("max_steps", 500)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    env      = WarehouseEnv(env_config)
    eval_env = WarehouseEnv(env_config)

    qmix = QMIX(qmix_config, env.obs_dim, env.action_dim, env.n_agents,
                device=device, total_timesteps=total_ts)
    if resume_path:
        qmix.load(resume_path)

    tracker = RewardTracker(n_agents=env.n_agents)

    print("=" * 60)
    print("QMIX Training")
    print("=" * 60)
    print(f"  Env        : {env_config['env']['name']}")
    print(f"  Agents     : {env.n_agents}")
    print(f"  Obs dim    : {env.obs_dim}")
    print(f"  Action dim : {env.action_dim}")
    print(f"  State dim  : {qmix.state_dim}")
    print(f"  Device     : {device}")
    print(f"  Total ts   : {total_ts:,}")
    print(f"  LR decay   : {qmix.lr_decay}")
    print(f"  Epsilon    : {qmix.epsilon_start} → {qmix.epsilon_end}")
    print("=" * 60)

    obs = env.reset()
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

        qmix.buffer.push(norm_obs, actions, np.array(rews), norm_next,
                         np.array(dones, dtype=np.float32), state, next_state)

        ep_rewards += rews
        ep_steps   += 1
        timestep   += 1
        obs         = next_obs

        # ---- Train ----
        stats = qmix.update()
        if stats:
            last_loss = stats["loss"]

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
                qmix.save(os.path.join(ckpt_dir, "qmix_best.pt"))

        # ---- Periodic save ----
        if timestep % save_int < 1:
            qmix.save(os.path.join(ckpt_dir, "qmix_latest.pt"))

    # ---- Final ----
    qmix.save(os.path.join(ckpt_dir, "qmix_latest.pt"))
    elapsed = time.time() - t_start
    print(f"\nTraining complete — {timestep:,} timesteps in {elapsed:.1f}s")
    print(f"Best eval reward: {best_reward:.4f}")

    os.makedirs(log_dir, exist_ok=True)
    tracker.save(os.path.join(log_dir, "qmix_train_rewards.json"), include_step_trace=False)
    tracker.save_csv(os.path.join(log_dir, "qmix_train_rewards.csv"))

    import json
    with open(os.path.join(log_dir, "qmix_eval_curve.json"), "w") as f:
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
    parser.add_argument("--qmix-config", default="configs/qmix_config.yaml")
    parser.add_argument("--resume",      default=None)
    args = parser.parse_args()

    env_config  = load_config(args.env_config)
    qmix_config = load_config(args.qmix_config)
    train(env_config, qmix_config, resume_path=args.resume)


if __name__ == "__main__":
    main()
