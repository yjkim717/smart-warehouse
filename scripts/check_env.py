"""
check_env.py — Installation check + environment sanity check

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --config configs/env_config.yaml
"""

import argparse
import sys
import yaml
import numpy as np

sys.path.insert(0, ".")
from src.env.warehouse_env import WarehouseEnv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_random_episode(env: WarehouseEnv, n_steps: int = 50) -> dict:
    obs = env.reset()
    total_rewards = [0.0] * env.n_agents
    collisions = 0
    deliveries = 0

    for step in range(n_steps):
        actions = [np.random.randint(env.action_dim) for _ in range(env.n_agents)]
        obs, rews, dones, info = env.step(actions)

        for i, r in enumerate(rews):
            total_rewards[i] += r
            if r > 0:
                deliveries += 1

        if all(dones):
            print(f"  Episode ended early at step {step + 1}")
            break

    return {
        "steps": step + 1,
        "total_rewards": total_rewards,
        "deliveries": deliveries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/env_config.yaml")
    args = parser.parse_args()

    print("=" * 50)
    print("Smart Warehouse — Environment Check")
    print("=" * 50)

    # 1. Import check
    try:
        import rware
        print(f"[OK] rware imported")
    except ImportError:
        print("[FAIL] rware not found — run: pip install rware")
        sys.exit(1)

    try:
        import gymnasium
        print(f"[OK] gymnasium imported (version: {gymnasium.__version__})")
    except ImportError:
        print("[FAIL] gymnasium not found — run: pip install gymnasium")
        sys.exit(1)

    # 2. Load config
    config = load_config(args.config)
    env_name = config["env"]["name"]
    print(f"\n[Config] env: {env_name}, n_agents: {config['env']['n_agents']}")

    # 3. Init env
    try:
        env = WarehouseEnv(config)
        print(f"[OK] WarehouseEnv created")
    except Exception as e:
        print(f"[FAIL] WarehouseEnv init failed: {e}")
        sys.exit(1)

    # 4. Print spaces
    print(f"\n  n_agents   : {env.n_agents}")
    print(f"  obs_dim    : {env.obs_dim}")
    print(f"  action_dim : {env.action_dim}")

    # 5. Reset
    try:
        obs = env.reset()
        assert len(obs) == env.n_agents, "obs count mismatch"
        assert obs[0].shape == (env.obs_dim,), f"obs shape mismatch: {obs[0].shape}"
        print(f"\n[OK] reset() — obs shape per agent: {obs[0].shape}")
    except Exception as e:
        print(f"[FAIL] reset() failed: {e}")
        sys.exit(1)

    # 6. Step
    try:
        actions = [0] * env.n_agents
        next_obs, rews, dones, info = env.step(actions)
        assert len(next_obs) == env.n_agents
        assert len(rews) == env.n_agents
        print(f"[OK] step() — rews: {rews}, dones: {dones}")
    except Exception as e:
        print(f"[FAIL] step() failed: {e}")
        sys.exit(1)

    # 7. Render
    try:
        frame = env.render()
        print(f"[OK] render() — frame shape: {frame.shape}")
    except Exception as e:
        print(f"[FAIL] render() failed: {e}")

    # 8. Random episode stats
    print(f"\nRunning 1 random episode (50 steps)...")
    stats = run_random_episode(env, n_steps=50)
    print(f"  Steps      : {stats['steps']}")
    print(f"  Rewards    : {stats['total_rewards']}")
    print(f"  Deliveries : {stats['deliveries']}")

    env.close()
    print("\n[DONE] All checks passed. Ready for B & C.")


if __name__ == "__main__":
    main()
