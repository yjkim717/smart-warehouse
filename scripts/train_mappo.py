"""
train_mappo.py — Train MAPPO agents in the warehouse environment.

Usage:
    python scripts/train_mappo.py
    python scripts/train_mappo.py --env-config configs/env_config.yaml --mappo-config configs/mappo_config.yaml
    python scripts/train_mappo.py --resume results/checkpoints/latest.pt
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np

sys.path.insert(0, ".")
from src.env.warehouse_env import WarehouseEnv
from src.algorithms import MAPPO
from src.analytics import RewardTracker


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(env: WarehouseEnv, mappo: MAPPO, n_episodes: int, max_steps: int) -> dict:
    """Run evaluation episodes with greedy actions (argmax), GRU-aware."""
    import torch

    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs = env.reset()
        mappo.reset_hidden()
        total_reward = 0.0

        for step in range(max_steps):
            raw_obs = np.stack(obs)
            norm_obs = mappo._normalize_obs(raw_obs, update=False)
            obs_t = torch.tensor(norm_obs, device=mappo.device)
            with torch.no_grad():
                if mappo.use_gru:
                    logits, mappo._gru_hidden = mappo.actor.forward(obs_t, mappo._gru_hidden)
                else:
                    logits = mappo.actor(obs_t)
                actions = logits.argmax(dim=-1).cpu().numpy()

            obs, rews, dones, _ = env.step(actions.tolist())
            total_reward += sum(rews)

            if all(dones):
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }


# ------------------------------------------------------------------
# Post-training: plots & GIF
# ------------------------------------------------------------------

def plot_training_curve(train_log: list, tracker, log_dir: str):
    """Generate training curve plot: eval reward over timesteps + random baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = "results/plots"
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Eval reward over training ---
    if train_log:
        ts = [e["timestep"] for e in train_log]
        means = [e["eval_mean_reward"] for e in train_log]
        stds = [e["eval_std_reward"] for e in train_log]

        means_arr = np.array(means)
        stds_arr = np.array(stds)

        axes[0].plot(ts, means, color="#1976D2", linewidth=1.5, label="MAPPO eval mean")
        axes[0].fill_between(
            ts, means_arr - stds_arr, means_arr + stds_arr,
            alpha=0.2, color="#1976D2",
        )

        # Random baseline reference line
        random_baseline_path = os.path.join(log_dir, "random_baseline_rewards.json")
        if os.path.exists(random_baseline_path):
            import json
            with open(random_baseline_path) as f:
                rb = json.load(f)
            rb_mean = rb["summary"]["team_total_reward"]["mean"]
            axes[0].axhline(rb_mean, color="#EF5350", linestyle="--", linewidth=1, label=f"Random baseline ({rb_mean:.3f})")

        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Team Total Reward")
        axes[0].set_title("Eval Reward over Training")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # --- Panel 2: Training episode rewards (rolling mean) ---
    if tracker.episodes:
        ep_rewards = [ep["team_total_reward"] for ep in tracker.episodes]
        axes[1].plot(ep_rewards, color="#90CAF9", alpha=0.3, linewidth=0.5, label="Per episode")

        window = min(100, max(1, len(ep_rewards) // 10))
        if len(ep_rewards) >= window:
            rolling = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
            axes[1].plot(
                range(window - 1, len(ep_rewards)),
                rolling, color="#1976D2", linewidth=1.5,
                label=f"Rolling mean ({window})",
            )

        axes[1].set_xlabel("Training Episode")
        axes[1].set_ylabel("Team Total Reward")
        axes[1].set_title("Training Episode Rewards")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    # --- Panel 3: Losses ---
    if train_log:
        ts = [e["timestep"] for e in train_log]
        axes[2].plot(ts, [e["policy_loss"] for e in train_log], label="Policy loss", color="#EF5350")
        axes[2].plot(ts, [e["value_loss"] for e in train_log], label="Value loss", color="#66BB6A")
        ax2 = axes[2].twinx()
        ax2.plot(ts, [e["entropy"] for e in train_log], label="Entropy", color="#AB47BC", linestyle="--")
        ax2.set_ylabel("Entropy", color="#AB47BC")

        axes[2].set_xlabel("Timestep")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("Training Losses & Entropy")
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "mappo_training_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[train] Training plot saved → {plot_path}")


def record_best_gif(checkpoint_path: str, env, env_config: dict):
    """Record a GIF using the best trained policy (MLP or GRU actor)."""
    import torch
    from src.algorithms.networks import Actor, GRUActor

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]

    use_gru = meta.get("use_gru", False)
    if use_gru:
        actor = GRUActor(meta["obs_dim"], meta["action_dim"], meta["hidden_dim"])
        hidden = actor.init_hidden(meta["n_agents"], torch.device("cpu"))
    else:
        actor = Actor(meta["obs_dim"], meta["action_dim"], meta["hidden_dim"], meta["n_layers"])
        hidden = None

    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_mean = ckpt.get("obs_rms", {}).get("mean", np.zeros(meta["obs_dim"]))
    obs_var = ckpt.get("obs_rms", {}).get("var", np.ones(meta["obs_dim"]))

    max_steps = env_config["env"].get("max_steps", 500)
    fps = env_config["env"]["record"]["fps"]
    output_dir = env_config["env"]["record"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    obs = env.reset()
    frames = [env.render()]

    for _ in range(max_steps):
        raw = np.stack(obs).astype(np.float64)
        norm = ((raw - obs_mean) / (np.sqrt(obs_var) + 1e-8)).astype(np.float32)
        obs_t = torch.tensor(norm)
        with torch.no_grad():
            if use_gru:
                # GRU actor: each agent independently (batch=1 per agent)
                actions = []
                new_hiddens = []
                for i in range(norm.shape[0]):
                    obs_i = torch.tensor(norm[i:i+1])
                    h_i = hidden[:, i:i+1, :]
                    logits_i, new_h_i = actor.forward(obs_i, h_i)
                    actions.append(logits_i.argmax(dim=-1).item())
                    new_hiddens.append(new_h_i)
                hidden = torch.cat(new_hiddens, dim=1)
            else:
                dist = torch.distributions.Categorical(logits=actor(obs_t))
                actions = dist.sample().numpy().tolist()

        obs, _, dones, _ = env.step(actions)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if all(dones):
            break

    try:
        import imageio
        gif_path = os.path.join(output_dir, "mappo_trained.gif")
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"[train] Trained policy GIF saved → {gif_path}  ({len(frames)} frames)")
    except ImportError:
        print("[train] imageio not installed — skipping GIF generation")


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train(env_config: dict, mappo_config: dict, resume_path: str | None = None):
    cfg = mappo_config["mappo"]
    total_timesteps = int(cfg["total_timesteps"])
    n_steps = cfg["n_steps"]
    eval_interval = cfg["eval_interval"]
    save_interval = cfg["save_interval"]
    log_interval = cfg["log_interval"]
    eval_episodes = cfg["eval_episodes"]
    ckpt_dir = cfg["checkpoint_dir"]
    log_dir = cfg["log_dir"]
    max_steps = env_config["env"].get("max_steps", 500)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    env = WarehouseEnv(env_config)
    eval_env = WarehouseEnv(env_config)

    mappo = MAPPO(mappo_config, env.obs_dim, env.action_dim, env.n_agents, device=device,
                  total_timesteps=total_timesteps)
    if resume_path:
        mappo.load(resume_path)

    tracker = RewardTracker(n_agents=env.n_agents)

    print("=" * 60)
    print("MAPPO Training")
    print("=" * 60)
    print(f"  Env            : {env_config['env']['name']}")
    print(f"  Agents         : {env.n_agents}")
    print(f"  Obs dim        : {env.obs_dim}")
    print(f"  Action dim     : {env.action_dim}")
    print(f"  Global obs dim : {mappo.global_obs_dim}")
    print(f"  Device         : {device}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Rollout length : {n_steps}")
    print(f"  LR decay       : {mappo.lr_decay}")
    print(f"  Entropy anneal : {mappo.entropy_coef_start:.3f} → {mappo.entropy_coef_end:.4f}")
    print(f"  Clip eps decay : {mappo.clip_eps_start:.2f} → {mappo.clip_eps_end:.2f}")
    print(f"  GRU actor      : {mappo.use_gru}")
    print("=" * 60)

    obs = env.reset()
    mappo.reset_hidden()
    ep_rewards = np.zeros(env.n_agents, dtype=np.float64)
    ep_steps = 0
    ep_count = 0
    best_eval_reward = -float("inf")

    recent_ep_rewards = []
    train_log = []
    t_start = time.time()
    timestep = 0

    while timestep < total_timesteps:
        # ---- Collect rollout ----
        for _ in range(n_steps):
            actions, log_probs, values, global_obs, norm_obs, hidden_np = mappo.select_actions(obs)
            next_obs, rews, dones, _ = env.step(actions.tolist())

            mappo.buffer.insert(
                norm_obs, global_obs, actions, log_probs, rews, dones, values, hidden_np,
            )

            ep_rewards += rews
            ep_steps += 1
            timestep += 1
            obs = next_obs

            if all(dones):
                mappo.reset_hidden()  # reset GRU state at episode boundary
                tracker.start_episode()
                # Retroactively log — we tracked rewards manually
                tracker._ep_steps = [[0.0] * env.n_agents]
                team_total = float(ep_rewards.sum())
                tracker._ep_steps = []
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
                ep_count += 1
                ep_rewards = np.zeros(env.n_agents, dtype=np.float64)
                ep_steps = 0
                obs = env.reset()

        # ---- Compute GAE & update ----
        next_values = mappo.get_values(obs)
        mappo.buffer.compute_returns(next_values, mappo.gamma, mappo.gae_lambda)
        losses = mappo.update()

        # ---- Step all scheduled hyperparameters ----
        mappo.step_schedulers(timestep)

        # ---- Logging ----
        if timestep % log_interval < n_steps and recent_ep_rewards:
            elapsed = time.time() - t_start
            fps = timestep / elapsed if elapsed > 0 else 0
            mean_r = np.mean(recent_ep_rewards[-50:])
            print(
                f"  t={timestep:>9,}  "
                f"episodes={ep_count:<6}  "
                f"mean_reward(50)={mean_r:>8.3f}  "
                f"p_loss={losses['policy_loss']:>7.4f}  "
                f"v_loss={losses['value_loss']:>7.4f}  "
                f"entropy={losses['entropy']:>.4f}  "
                f"ent_coef={mappo.entropy_coef:.4f}  "
                f"clip_eps={mappo.clip_eps:.3f}  "
                f"lr={mappo.get_lr():.2e}  "
                f"fps={fps:>6.0f}"
            )

        # ---- Evaluation ----
        if timestep % eval_interval < n_steps:
            eval_stats = evaluate(eval_env, mappo, eval_episodes, max_steps)
            print(
                f"  [eval] t={timestep:>9,}  "
                f"reward={eval_stats['mean_reward']:.3f} ± {eval_stats['std_reward']:.3f}  "
                f"max={eval_stats['max_reward']:.1f}  "
                f"length={eval_stats['mean_length']:.0f}"
            )

            train_log.append({
                "timestep": timestep,
                "eval_mean_reward": round(eval_stats["mean_reward"], 4),
                "eval_std_reward": round(eval_stats["std_reward"], 4),
                "eval_max_reward": round(eval_stats["max_reward"], 4),
                "policy_loss": round(losses["policy_loss"], 6),
                "value_loss": round(losses["value_loss"], 6),
                "entropy": round(losses["entropy"], 6),
                "entropy_coef": round(mappo.entropy_coef, 6),
                "clip_eps": round(mappo.clip_eps, 4),
                "lr_actor": round(mappo.get_lr(), 8),
            })

            if eval_stats["mean_reward"] > best_eval_reward:
                best_eval_reward = eval_stats["mean_reward"]
                mappo.save(os.path.join(ckpt_dir, "best_model.pt"))

        # ---- Periodic save ----
        if timestep % save_interval < n_steps:
            mappo.save(os.path.join(ckpt_dir, "latest.pt"))

    # ---- Final save ----
    mappo.save(os.path.join(ckpt_dir, "latest.pt"))

    elapsed = time.time() - t_start
    print(f"\nTraining complete — {timestep:,} timesteps in {elapsed:.1f}s")
    print(f"Best eval reward: {best_eval_reward:.4f}")

    # Save training reward log
    os.makedirs(log_dir, exist_ok=True)
    tracker.save(os.path.join(log_dir, "mappo_train_rewards.json"), include_step_trace=False)
    tracker.save_csv(os.path.join(log_dir, "mappo_train_rewards.csv"))

    # Save eval curve
    import json
    eval_log_path = os.path.join(log_dir, "mappo_eval_curve.json")
    with open(eval_log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"[train] Eval curve saved → {eval_log_path}")

    # ---- Generate training curve plot ----
    plot_training_curve(train_log, tracker, log_dir)

    # ---- Record GIF of best policy ----
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        record_best_gif(best_ckpt, eval_env, env_config)

    env.close()
    eval_env.close()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--mappo-config", default="configs/mappo_config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    env_config = load_config(args.env_config)
    mappo_config = load_config(args.mappo_config)

    train(env_config, mappo_config, resume_path=args.resume)


if __name__ == "__main__":
    main()
