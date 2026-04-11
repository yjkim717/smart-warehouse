"""
record_gif.py — Record environment rendering as a GIF

What this script does:
    1. Loads the warehouse environment from configs/env_config.yaml
    2. Runs one episode using a semi-greedy random policy:
         - Each agent moves randomly by default
         - If an agent is empty-handed and standing on a shelf → picks it up (action 4)
         - If an agent is carrying a shelf and standing on a packing station (P) → drops it (action 4)
         - After dropping, the agent has no shelf (empty-handed) and resumes random movement
    3. Captures an RGB frame after every step
    4. Saves all frames as an animated GIF to results/videos/

Usage:
    # Random policy (default)
    python scripts/record_gif.py

    # Use a trained policy checkpoint
    python scripts/record_gif.py --checkpoint results/checkpoints/best_model.pt

    # Override config or step count
    python scripts/record_gif.py --config configs/env_config.yaml --steps 200
"""

import argparse
import os
import sys
import yaml
import numpy as np

sys.path.insert(0, ".")
from src.env.warehouse_env import WarehouseEnv
from src.analytics import RewardTracker


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def random_policy(obs: list, action_dim: int, env: "WarehouseEnv" = None) -> list:
    """
    Random actions with two greedy overrides:
      - Drop shelf (action 4) when carrying and standing on a packing station.
      - Pick up shelf (action 4) when empty-handed and standing on a shelf.
    """
    ACTION_INTERACT = 4
    actions = [np.random.randint(action_dim) for _ in obs]

    if env is not None:
        u = env._env.unwrapped
        goal_set = {(gc, gr) for gc, gr in u.goals}     # packing station positions (col, row)
        shelf_set = {(s.x, s.y) for s in u.shelfs}      # shelf positions (col, row)

        for i, agent in enumerate(u.agents):
            pos = (agent.x, agent.y)
            if agent.carrying_shelf and pos in goal_set:
                actions[i] = ACTION_INTERACT             # drop shelf at packing station
            elif agent.carrying_shelf:
                actions[i] = np.random.randint(4)       # carrying but not at goal — move only, no accidental drop
            elif not agent.carrying_shelf and pos in goal_set:
                actions[i] = np.random.randint(4)       # move away from packing station (actions 0-3 only, no INTERACT)
            elif not agent.carrying_shelf and pos in shelf_set:
                actions[i] = ACTION_INTERACT             # pick up new shelf

    return actions


def load_trained_policy(checkpoint_path: str):
    """Load MAPPO actor from checkpoint and return a greedy policy function."""
    import torch
    from src.algorithms.networks import Actor

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]

    actor = Actor(meta["obs_dim"], meta["action_dim"], meta["hidden_dim"], meta["n_layers"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    # Restore observation normalization stats
    obs_mean = ckpt.get("obs_rms", {}).get("mean", np.zeros(meta["obs_dim"]))
    obs_var = ckpt.get("obs_rms", {}).get("var", np.ones(meta["obs_dim"]))

    def policy_fn(obs, action_dim, env=None):
        raw = np.stack(obs).astype(np.float64)
        norm = ((raw - obs_mean) / (np.sqrt(obs_var) + 1e-8)).astype(np.float32)
        obs_t = torch.tensor(norm)
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=actor(obs_t))
            actions = dist.sample()
        return actions.numpy().tolist()

    return policy_fn


def record(env: WarehouseEnv, policy_fn, n_steps: int, tracker: RewardTracker = None) -> list:
    """Run one episode, collect RGB frames, and optionally track rewards."""
    frames = []
    obs = env.reset()

    if tracker is not None:
        tracker.start_episode()

    # Capture initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    for _ in range(n_steps):
        actions = policy_fn(obs, env.action_dim, env)
        obs, rews, dones, _ = env.step(actions)

        if tracker is not None:
            tracker.record_step(rews)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if all(dones):
            break

    if tracker is not None:
        tracker.end_episode()
        ep = tracker.episodes[-1]
        print(
            f"[reward] episode total={ep['team_total_reward']:.3f}  "
            f"steps={ep['n_steps']}  "
            f"per_step={ep['team_reward_per_step']:.5f}"
        )

    return frames


def save_gif(frames: list, output_path: str, fps: int):
    try:
        import imageio
    except ImportError:
        print("[FAIL] imageio not found — run: pip install imageio")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"[OK] GIF saved → {output_path}  ({len(frames)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint")
    parser.add_argument("--steps", type=int, default=None, help="Override max steps for recording")
    args = parser.parse_args()

    config = load_config(args.config)
    rec_cfg = config["env"]["record"]

    n_steps = args.steps or config["env"].get("max_steps", 500)
    fps = rec_cfg["fps"]
    output_dir = rec_cfg["output_dir"]

    # Pick policy
    if args.checkpoint:
        print(f"Loading trained policy from {args.checkpoint} ...")
        policy_fn = load_trained_policy(args.checkpoint)
        filename = "trained_policy.gif"
    else:
        print("Using random policy (no checkpoint provided)")
        policy_fn = random_policy
        filename = rec_cfg["filename"]

    output_path = os.path.join(output_dir, filename)

    # Run
    env = WarehouseEnv(config)
    tracker = RewardTracker(n_agents=env.n_agents)

    print(f"Recording {n_steps} steps in {config['env']['name']} ...")
    frames = record(env, policy_fn, n_steps, tracker=tracker)
    env.close()

    print(f"Captured {len(frames)} frames")
    save_gif(frames, output_path, fps)

    # Save reward log alongside the GIF
    log_dir = "results/logs"
    policy_tag = "trained" if args.checkpoint else "random"
    tracker.save(os.path.join(log_dir, f"{policy_tag}_policy_rewards.json"))
    tracker.save_csv(os.path.join(log_dir, f"{policy_tag}_policy_rewards.csv"))


if __name__ == "__main__":
    main()
