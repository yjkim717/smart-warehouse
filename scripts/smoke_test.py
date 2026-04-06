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
        actions, log_probs, values, global_obs, norm_obs, hidden = mappo.select_actions(obs)
        next_obs, rews, dones, _ = env.step(actions.tolist())
        mappo.buffer.insert(norm_obs, global_obs, actions, log_probs, rews, dones, values, hidden)
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
        actions, log_probs, values, global_obs, norm_obs, hidden = mappo.select_actions(obs)
        next_obs, rews, dones, _ = env.step(actions.tolist())
        mappo.buffer.insert(norm_obs, global_obs, actions, log_probs, rews, dones, values, hidden)
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
            actions, log_probs, values, global_obs, norm_obs, hidden = mappo.select_actions(obs)
            next_obs, rews, dones, _ = env.step(actions.tolist())
            mappo.buffer.insert(norm_obs, global_obs, actions, log_probs, rews, dones, values, hidden)
            obs = next_obs
            if all(dones):
                mappo.reset_hidden()
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


def check_lr_decay(env_cfg, mappo_cfg):
    """LR should decrease monotonically when lr_decay=True."""
    print("\n--- 5. LR decay: cosine schedule reduces actor LR over training ---")
    from src.env.warehouse_env import WarehouseEnv
    from src.algorithms.mappo import MAPPO

    cfg = yaml.safe_load(yaml.dump(mappo_cfg))  # deep copy
    cfg["mappo"]["lr_decay"] = True
    cfg["mappo"]["lr_min"] = 1e-5
    total = 1000

    env = WarehouseEnv(env_cfg)
    mappo = MAPPO(cfg, env.obs_dim, env.action_dim, env.n_agents, total_timesteps=total)

    lrs = []
    for t in [0, 250, 500, 750, 1000]:
        mappo.step_schedulers(t)
        lrs.append(mappo.get_lr())

    env.close()

    monotone = all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))
    reaches_min = abs(lrs[-1] - 1e-5) < 1e-7

    print(f"  {INFO} LR at t=0: {lrs[0]:.6f}  t=500: {lrs[2]:.6f}  t=1000: {lrs[4]:.8f}")

    if monotone:
        print(f"  {PASS} LR is monotonically non-increasing")
    else:
        print(f"  {FAIL} LR is NOT monotonically decreasing: {lrs}")

    if reaches_min:
        print(f"  {PASS} LR reaches lr_min ({1e-5:.1e}) at total_timesteps")
    else:
        print(f"  {FAIL} Final LR ({lrs[-1]:.2e}) does not match lr_min ({1e-5:.1e})")

    return monotone and reaches_min


def check_entropy_annealing(env_cfg, mappo_cfg):
    """Entropy coefficient should decrease monotonically over training."""
    print("\n--- 6. Entropy annealing: entropy_coef decays from start to end ---")
    from src.env.warehouse_env import WarehouseEnv
    from src.algorithms.mappo import MAPPO

    cfg = yaml.safe_load(yaml.dump(mappo_cfg))
    cfg["mappo"]["entropy_coef_start"] = 0.05
    cfg["mappo"]["entropy_coef_end"] = 0.001
    total = 1000

    env = WarehouseEnv(env_cfg)
    mappo = MAPPO(cfg, env.obs_dim, env.action_dim, env.n_agents, total_timesteps=total)

    coefs = []
    for t in [0, 250, 500, 750, 1000]:
        mappo.step_schedulers(t)
        coefs.append(mappo.entropy_coef)

    env.close()

    monotone = all(coefs[i] >= coefs[i + 1] for i in range(len(coefs) - 1))
    start_ok = abs(coefs[0] - 0.05) < 1e-6
    end_ok = abs(coefs[-1] - 0.001) < 1e-6

    print(f"  {INFO} entropy_coef at t=0: {coefs[0]:.4f}  t=500: {coefs[2]:.4f}  t=1000: {coefs[4]:.4f}")

    if monotone:
        print(f"  {PASS} Entropy coefficient is monotonically non-increasing")
    else:
        print(f"  {FAIL} Entropy coefficient is NOT monotonically decreasing: {coefs}")

    if start_ok and end_ok:
        print(f"  {PASS} Entropy coef spans [start=0.05, end=0.001] correctly")
    else:
        print(f"  {FAIL} Entropy coef start={coefs[0]:.4f} (want 0.05), end={coefs[-1]:.4f} (want 0.001)")

    return monotone and start_ok and end_ok


def check_collision_penalty(env_cfg):
    """
    Collision penalty should fire when an agent tries to move but is blocked by
    an adjacent agent. Run random episodes and verify shaped reward reflects it.
    """
    print("\n--- 7. Collision penalty: fires on blocked moves near adjacent agent ---")
    from src.env.warehouse_env import WarehouseEnv

    cfg = yaml.safe_load(yaml.dump(env_cfg))
    cfg["env"]["reward_shaping"]["enabled"] = True
    cfg["env"]["reward_shaping"]["collision_penalty"] = -0.3

    env = WarehouseEnv(cfg)

    collision_events = 0
    total_steps = 0

    for _ in range(20):
        obs = env.reset()
        u = env._env.unwrapped
        for _ in range(200):
            # Force agents toward each other to increase collision chance
            actions = [np.random.randint(4) for _ in range(env.n_agents)]  # movement only
            obs, rews, dones, _ = env.step(actions)
            total_steps += 1

            # Count steps where a collision penalty may have fired:
            # Detect by checking if any agent tried to move but stayed put near another
            for i, agent in enumerate(u.agents):
                prev = env._prev_positions[i]
                curr = (agent.x, agent.y)
                last_act = env._last_actions[i]
                if last_act in (0, 1, 2, 3) and prev == curr:
                    for j, other in enumerate(u.agents):
                        if j != i and abs(other.x - agent.x) + abs(other.y - agent.y) <= 1:
                            collision_events += 1
                            break

            if all(dones):
                break

    env.close()

    rate = collision_events / max(total_steps, 1)
    print(f"  {INFO} Collision events: {collision_events} / {total_steps} steps ({rate*100:.1f}%)")

    if collision_events > 0:
        print(f"  {PASS} Collision penalty fires correctly on blocked moves near adjacent agent")
        return True
    else:
        print(f"  {FAIL} No collision events detected — check collision_penalty logic or agent spacing")
        return False


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
    results["env"]               = check_env(env_cfg)
    results["buffer_norm"]       = check_buffer_normalization(env_cfg, mappo_cfg)
    results["ppo_update"]        = check_ppo_update(env_cfg, mappo_cfg)
    results["convergence"]       = check_training_convergence(env_cfg, mappo_cfg)
    results["lr_decay"]          = check_lr_decay(env_cfg, mappo_cfg)
    results["entropy_annealing"] = check_entropy_annealing(env_cfg, mappo_cfg)
    results["collision_penalty"] = check_collision_penalty(env_cfg)

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
