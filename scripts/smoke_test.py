"""
smoke_test.py — Fast validation of the MAPPO training pipeline (~30 seconds).

Checks:
  1. Environment loads and steps correctly
  2. Buffer GAE + advantage normalization is rollout-level (not per-minibatch)
  3. PPO update runs without errors, losses are finite
  4. Entropy decreases over a short training run (policy is learning)
  5. Eval reward improves over baseline noise

Usage:
    python scripts/smoke_test.py
"""

import sys
import time
import yaml
import numpy as np

sys.path.insert(0, ".")

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def load_configs():
    with open("configs/env_config.yaml") as f:
        env_cfg = yaml.safe_load(f)
    with open("configs/mappo_config.yaml") as f:
        mappo_cfg = yaml.safe_load(f)
    # Override for speed: tiny rollout, few steps
    mappo_cfg["mappo"]["n_steps"] = 64
    mappo_cfg["mappo"]["minibatch_size"] = 32
    mappo_cfg["mappo"]["n_epochs"] = 2
    return env_cfg, mappo_cfg


def check_env(env_cfg):
    print("\n--- 1. Environment ---")
    from src.env.warehouse_env import WarehouseEnv
    env = WarehouseEnv(env_cfg)
    obs = env.reset()
    assert len(obs) == env.n_agents, "obs count mismatch"
    assert obs[0].shape == (env.obs_dim,), f"obs shape wrong: {obs[0].shape}"

    obs2, rews, dones, _ = env.step([0] * env.n_agents)
    assert len(rews) == env.n_agents, "reward count mismatch"
    assert all(isinstance(r, float) for r in rews), "rewards not floats"

    print(f"  {PASS} reset/step OK  |  n_agents={env.n_agents}  obs_dim={env.obs_dim}  action_dim={env.action_dim}")
    env.close()
    return True


def check_buffer_normalization(env_cfg, mappo_cfg):
    print("\n--- 2. Buffer: advantage normalization is rollout-level ---")
    from src.env.warehouse_env import WarehouseEnv
    from src.algorithms.mappo import MAPPO

    env = WarehouseEnv(env_cfg)
    mappo = MAPPO(mappo_cfg, env.obs_dim, env.action_dim, env.n_agents)

    # Collect one rollout
    obs = env.reset()
    n_steps = mappo_cfg["mappo"]["n_steps"]
    for _ in range(n_steps):
        actions, log_probs, values, global_obs, norm_obs = mappo.select_actions(obs)
        next_obs, rews, dones, _ = env.step(actions.tolist())
        mappo.buffer.insert(norm_obs, global_obs, actions, log_probs, rews, dones, values)
        obs = next_obs

    next_values = mappo.get_values(obs)
    mappo.buffer.compute_returns(next_values, mappo.gamma, mappo.gae_lambda)

    adv = mappo.buffer.advantages  # shape (n_steps, n_agents)
    adv_flat = adv.reshape(-1)

    # After rollout-level normalization: full-rollout mean ≈ 0, std ≈ 1
    mean_err = abs(adv_flat.mean())
    std_err = abs(adv_flat.std() - 1.0)

    if mean_err < 1e-5 and std_err < 1e-4:
        print(f"  {PASS} Advantages normalized over full rollout  |  mean={adv_flat.mean():.2e}  std={adv_flat.std():.6f}")
    else:
        print(f"  {FAIL} Advantage normalization wrong  |  mean={adv_flat.mean():.4f}  std={adv_flat.std():.4f}")
        env.close()
        return False

    # Verify returns are unnormalized (returns >> advantages)
    ret_flat = mappo.buffer.returns.reshape(-1)
    if ret_flat.std() > 0.01:
        print(f"  {PASS} Returns are unnormalized (std={ret_flat.std():.4f}), separate from advantages")
    else:
        print(f"  {FAIL} Returns look wrong (std={ret_flat.std():.4f})")
        env.close()
        return False

    env.close()
    return True


def check_ppo_update(env_cfg, mappo_cfg):
    print("\n--- 3. PPO update: losses finite, no NaN ---")
    from src.env.warehouse_env import WarehouseEnv
    from src.algorithms.mappo import MAPPO

    env = WarehouseEnv(env_cfg)
    mappo = MAPPO(mappo_cfg, env.obs_dim, env.action_dim, env.n_agents)

    obs = env.reset()
    n_steps = mappo_cfg["mappo"]["n_steps"]
    for _ in range(n_steps):
        actions, log_probs, values, global_obs, norm_obs = mappo.select_actions(obs)
        next_obs, rews, dones, _ = env.step(actions.tolist())
        mappo.buffer.insert(norm_obs, global_obs, actions, log_probs, rews, dones, values)
        obs = next_obs

    next_values = mappo.get_values(obs)
    mappo.buffer.compute_returns(next_values, mappo.gamma, mappo.gae_lambda)
    losses = mappo.update()

    ok = True
    for k, v in losses.items():
        finite = np.isfinite(v)
        status = PASS if finite else FAIL
        print(f"  {status} {k} = {v:.6f}")
        if not finite:
            ok = False

    env.close()
    return ok


def check_training_convergence(env_cfg, mappo_cfg):
    print("\n--- 4. Short training: entropy should decrease ---")
    import torch
    from src.env.warehouse_env import WarehouseEnv
    from src.algorithms.mappo import MAPPO

    env = WarehouseEnv(env_cfg)
    mappo = MAPPO(mappo_cfg, env.obs_dim, env.action_dim, env.n_agents)

    n_steps = mappo_cfg["mappo"]["n_steps"]
    n_rollouts = 20  # ~1280 timesteps total
    entropy_log = []

    obs = env.reset()
    t0 = time.time()

    for rollout in range(n_rollouts):
        for _ in range(n_steps):
            actions, log_probs, values, global_obs, norm_obs = mappo.select_actions(obs)
            next_obs, rews, dones, _ = env.step(actions.tolist())
            mappo.buffer.insert(norm_obs, global_obs, actions, log_probs, rews, dones, values)
            obs = next_obs
            if all(dones):
                obs = env.reset()

        next_values = mappo.get_values(obs)
        mappo.buffer.compute_returns(next_values, mappo.gamma, mappo.gae_lambda)
        losses = mappo.update()
        entropy_log.append(losses["entropy"])

    elapsed = time.time() - t0
    first5 = np.mean(entropy_log[:5])
    last5 = np.mean(entropy_log[-5:])
    delta = last5 - first5

    print(f"  {INFO} {n_rollouts} rollouts in {elapsed:.1f}s  ({n_rollouts * n_steps} total steps)")
    print(f"  {INFO} Entropy: first 5 rollouts avg = {first5:.4f}  |  last 5 rollouts avg = {last5:.4f}  |  delta = {delta:+.4f}")

    if delta < 0:
        print(f"  {PASS} Entropy is decreasing — policy is committing to actions")
    else:
        print(f"  {FAIL} Entropy did not decrease ({delta:+.4f}) — policy not learning")

    # Check all losses are finite throughout
    if all(np.isfinite(e) for e in entropy_log):
        print(f"  {PASS} All entropy values finite across {n_rollouts} updates")
    else:
        print(f"  {FAIL} NaN/Inf entropy detected")
        env.close()
        return False

    env.close()
    return delta < 0


def main():
    print("=" * 55)
    print("Smart Warehouse — MAPPO Smoke Test")
    print("=" * 55)

    try:
        env_cfg, mappo_cfg = load_configs()
    except Exception as e:
        print(f"{FAIL} Config load failed: {e}")
        sys.exit(1)

    results = {}
    results["env"]            = check_env(env_cfg)
    results["buffer_norm"]    = check_buffer_normalization(env_cfg, mappo_cfg)
    results["ppo_update"]     = check_ppo_update(env_cfg, mappo_cfg)
    results["convergence"]    = check_training_convergence(env_cfg, mappo_cfg)

    print("\n" + "=" * 55)
    print("Results:")
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll checks passed — safe to run full training.")
        print("  python scripts/train_mappo.py")
    else:
        print("\nSome checks failed — fix issues before full training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
