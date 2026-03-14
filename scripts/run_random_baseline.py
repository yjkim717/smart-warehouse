"""
run_random_baseline.py — Run random agent over N episodes and measure reward.

No GIF recording — pure reward measurement for baseline reporting.

Usage:
    python scripts/run_random_baseline.py
    python scripts/run_random_baseline.py --episodes 50
    python scripts/run_random_baseline.py --config configs/env_config.yaml --episodes 100
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


def random_policy(obs: list, action_dim: int, env: WarehouseEnv = None) -> list:
    ACTION_INTERACT = 4
    actions = [np.random.randint(action_dim) for _ in obs]

    if env is not None:
        u = env._env.unwrapped
        goal_set = {(gc, gr) for gc, gr in u.goals}
        shelf_set = {(s.x, s.y) for s in u.shelfs}

        for i, agent in enumerate(u.agents):
            pos = (agent.x, agent.y)
            if agent.carrying_shelf and pos in goal_set:
                actions[i] = ACTION_INTERACT
            elif agent.carrying_shelf:
                actions[i] = np.random.randint(4)
            elif not agent.carrying_shelf and pos in goal_set:
                actions[i] = np.random.randint(4)
            elif not agent.carrying_shelf and pos in shelf_set:
                actions[i] = ACTION_INTERACT

    return actions


def run_episode(env: WarehouseEnv, tracker: RewardTracker, max_steps: int):
    obs = env.reset()
    tracker.start_episode()

    for _ in range(max_steps):
        actions = random_policy(obs, env.action_dim, env)
        obs, rews, dones, _ = env.step(actions)
        tracker.record_step(rews)
        if all(dones):
            break

    tracker.end_episode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_config.yaml")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes to run")
    parser.add_argument("--log-dir", default="results/logs")
    args = parser.parse_args()

    config = load_config(args.config)
    max_steps = config["env"].get("max_steps", 500)

    env = WarehouseEnv(config)
    tracker = RewardTracker(n_agents=env.n_agents)

    print(f"Running random baseline: {args.episodes} episodes, max {max_steps} steps each")
    print(f"Env: {config['env']['name']}  |  Agents: {env.n_agents}\n")

    for ep in range(args.episodes):
        run_episode(env, tracker, max_steps)
        ep_data = tracker.episodes[-1]
        print(
            f"  ep {ep+1:3d}/{args.episodes}  "
            f"steps={ep_data['n_steps']:4d}  "
            f"team_reward={ep_data['team_total_reward']:.3f}  "
            f"per_step={ep_data['team_reward_per_step']:.5f}"
        )

    env.close()

    # Print summary
    summary = tracker.summary()
    print("\n--- Random Baseline Summary ---")
    print(f"  Episodes : {summary['n_episodes']}")
    print(f"  Team total reward  — mean: {summary['team_total_reward']['mean']:.4f}  "
          f"std: {summary['team_total_reward']['std']:.4f}  "
          f"[{summary['team_total_reward']['min']:.4f}, {summary['team_total_reward']['max']:.4f}]")
    print(f"  Episode length     — mean: {summary['episode_length']['mean']:.1f}  "
          f"std: {summary['episode_length']['std']:.1f}")
    for i, v in enumerate(summary["per_agent_mean_total"]):
        print(f"  Agent {i} mean total reward: {v:.4f}")

    # Save
    os.makedirs(args.log_dir, exist_ok=True)
    tracker.save(
        os.path.join(args.log_dir, "random_baseline_rewards.json"),
        include_step_trace=False,
    )
    tracker.save_csv(os.path.join(args.log_dir, "random_baseline_rewards.csv"))


if __name__ == "__main__":
    main()
