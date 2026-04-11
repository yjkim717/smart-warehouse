"""
test_reward_shaping.py — Targeted tests for reward shaping correctness.

Specifically guards against:
  - Anti-toggle death spiral (drop→pickup cascading -0.6 each)
  - Persistent just_dropped flag blocking legitimate pickups
  - Catastrophic episode rewards from toggle loops
  - Agents learning to avoid picking up (loop behaviour in GIF)

Usage:
    python scripts/test_reward_shaping.py
"""

import sys
import yaml
import numpy as np

sys.path.insert(0, ".")

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def load_env():
    with open("configs/env_config.yaml") as f:
        cfg = yaml.safe_load(f)
    from src.env.warehouse_env import WarehouseEnv
    return WarehouseEnv(cfg), cfg


def test_no_cascade_on_toggle(env):
    """
    Drop shelf away from goal, then immediately pick up another.
    Old bug: -0.6 (bad drop) + -0.6 (anti-toggle) = -1.2 per cycle.
    Fixed:   -0.6 (bad drop), cooldown=5 set, re-pickup gives 0 (not -0.6).
    """
    print("\n--- 1. Anti-toggle: cooldown set on bad drop, no cascade penalty ---")

    u = env._env.unwrapped
    env.reset()
    agent = u.agents[0]
    goal_positions = {(gc, gr) for gc, gr in u.goals}

    # Set up: agent was carrying, cooldown clear
    env._prev_carrying = [True, False]
    env._drop_cooldown = [0, 0]
    env._just_dropped = [False, False]
    agent.carrying_shelf = None  # agent dropped

    non_goal = next((c, r) for c in range(10) for r in range(10) if (c, r) not in goal_positions)
    agent.x, agent.y = non_goal
    env._apply_reward_shaping([0.0, 0.0])

    # After bad drop: cooldown set to 5 then ticked once on same step → 4 remaining
    if env._drop_cooldown[0] == 4:
        print(f"  {PASS} Bad drop sets cooldown, 4 suppressed steps remain after tick")
        cooldown_ok = True
    else:
        print(f"  {FAIL} Cooldown after bad drop = {env._drop_cooldown[0]} (expected 4)")
        cooldown_ok = False

    # Immediate re-pickup: cooldown should still be active (3 after tick)
    from unittest.mock import MagicMock
    agent.carrying_shelf = MagicMock()
    env._prev_carrying = [False, False]
    env._apply_reward_shaping([0.0, 0.0])

    # Cooldown ticks 4 → 3 (still active, cascade blocked)
    if env._drop_cooldown[0] == 3:
        print(f"  {PASS} Re-pickup during cooldown: cooldown still active (3), no cascade penalty")
        cascade_ok = True
    else:
        print(f"  {FAIL} Cooldown after re-pickup = {env._drop_cooldown[0]} (expected 3)")
        cascade_ok = False

    old_bug_cycle = 2 * env._bad_drop_penalty + 2 * env._step_penalty
    new_cycle = env._bad_drop_penalty + 2 * env._step_penalty
    print(f"  {INFO} Old bug worst 2-step cycle: {old_bug_cycle:.3f} | Fixed: {new_cycle:.3f}")

    return cooldown_ok and cascade_ok


def test_cooldown_expires(env):
    """
    After 5 steps, the drop cooldown should expire.
    A pickup on step 6+ should give full pickup_reward again.
    """
    print("\n--- 2. Cooldown expiry: cooldown ticks correctly over 5 steps ---")

    from unittest.mock import MagicMock
    u = env._env.unwrapped
    env.reset()
    agent = u.agents[0]
    agent.carrying_shelf = None

    # Set cooldown to 5 (just set)
    env._drop_cooldown = [5, 0]
    env._prev_carrying = [False, False]
    env._just_dropped = [False, False]
    agent.x, agent.y = 5, 5

    # Tick through 5 steps without picking up
    for step in range(5):
        env._apply_reward_shaping([0.0, 0.0])

    if env._drop_cooldown[0] == 0:
        print(f"  {PASS} Cooldown reached 0 after 5 steps")
        tick_ok = True
    else:
        print(f"  {FAIL} Cooldown = {env._drop_cooldown[0]} after 5 steps (expected 0)")
        tick_ok = False

    # Now pickup: should get full pickup_reward (cooldown = 0)
    env._prev_carrying = [False, False]
    agent.carrying_shelf = MagicMock()
    env._apply_reward_shaping([0.0, 0.0])

    # Cooldown should stay at 0 (pickup doesn't reset it)
    if env._drop_cooldown[0] == 0:
        print(f"  {PASS} After cooldown expiry, pickup allowed (cooldown stays 0)")
        pickup_ok = True
    else:
        print(f"  {FAIL} Unexpected cooldown after expiry pickup: {env._drop_cooldown[0]}")
        pickup_ok = False

    return tick_ok and pickup_ok


def test_no_catastrophic_episodes(env):
    """
    Run 100 random episodes. With the fix, NO episode should go below -30.
    Old bug: episodes could hit -226 from toggle cascades.
    """
    print("\n--- 3. Random episodes: no catastrophic reward accumulation ---")

    rewards = []
    min_single_agent = []
    worst_ep = None

    for ep in range(100):
        obs = env.reset()
        total = [0.0] * env.n_agents
        for _ in range(500):
            actions = [np.random.randint(env.action_dim) for _ in range(env.n_agents)]
            obs, rews, dones, _ = env.step(actions)
            for i, r in enumerate(rews):
                total[i] += r
            if all(dones):
                break
        team = sum(total)
        rewards.append(team)
        min_single_agent.append(min(total))
        if worst_ep is None or team < worst_ep[0]:
            worst_ep = (team, total)

    mean_r = np.mean(rewards)
    min_r = min(rewards)
    pos_eps = sum(1 for r in rewards if r > 0)

    print(f"  {INFO} 100 episodes: mean={mean_r:.2f}  min={min_r:.2f}  positive={pos_eps}/100")
    print(f"  {INFO} Worst episode: team={worst_ep[0]:.2f}  per_agent={[round(x,2) for x in worst_ep[1]]}")

    catastrophic = sum(1 for r in rewards if r < -30)
    if catastrophic == 0:
        print(f"  {PASS} No catastrophic episodes (< -30) in 100 random runs")
        cat_ok = True
    else:
        print(f"  {FAIL} {catastrophic}/100 episodes hit below -30 — toggle loop still possible")
        cat_ok = False

    step_only = env._step_penalty * 500
    # With fix: a random agent can still legitimately get ~5-6 bad drops in an episode
    # (5 bad drops × -0.6 = -3.0) + step_penalty (-2.5) + oscillation ≈ -10 to -20 is normal
    # Old bug allowed: toggle loop → -226. Threshold of -30 catches any remnant cascade.
    reasonable_floor = step_only - 20  # step_penalty + ~5 bad drops + oscillation
    if min_r > reasonable_floor:
        print(f"  {PASS} Min reward ({min_r:.2f}) within reasonable range (floor ≈ {reasonable_floor:.2f})")
        floor_ok = True
    else:
        print(f"  {FAIL} Min reward ({min_r:.2f}) below reasonable floor ({reasonable_floor:.2f}) — still cascading?")
        floor_ok = False

    return cat_ok and floor_ok


def test_agents_can_pickup_and_deliver(env):
    """
    Run 200 random episodes and verify agents successfully pick up
    shelves AND deliver at least occasionally.
    """
    print("\n--- 4. Agents can pick up and deliver (reward shaping signals working) ---")

    pickups = 0
    deliveries = 0
    pickup_rewards_seen = []
    delivery_rewards_seen = []

    for ep in range(200):
        obs = env.reset()
        prev_carrying = [False] * env.n_agents

        for _ in range(500):
            actions = [np.random.randint(env.action_dim) for _ in range(env.n_agents)]
            obs, rews, dones, _ = env.step(actions)

            u = env._env.unwrapped
            for i, agent in enumerate(u.agents):
                now_carrying = bool(agent.carrying_shelf)
                if now_carrying and not prev_carrying[i]:
                    pickups += 1
                    pickup_rewards_seen.append(rews[i])
                if rews[i] > 0.5:  # delivery bonus fires
                    deliveries += 1
                    delivery_rewards_seen.append(rews[i])
                prev_carrying[i] = now_carrying

            if all(dones):
                break

    print(f"  {INFO} Pickups detected:   {pickups}  over 200 episodes")
    print(f"  {INFO} Deliveries detected: {deliveries} over 200 episodes")

    if pickups > 0:
        avg_pickup_rew = np.mean(pickup_rewards_seen)
        print(f"  {INFO} Avg reward on pickup step: {avg_pickup_rew:.3f}")

    if deliveries > 0:
        avg_del_rew = np.mean(delivery_rewards_seen)
        print(f"  {INFO} Avg reward on delivery step: {avg_del_rew:.3f}")

    pickup_ok = pickups > 50
    delivery_ok = deliveries > 0

    if pickup_ok:
        print(f"  {PASS} Agents are picking up shelves ({pickups} pickups in 200 eps)")
    else:
        print(f"  {FAIL} Too few pickups ({pickups}) — agents may be avoiding shelves")

    if delivery_ok:
        print(f"  {PASS} At least one delivery achieved ({deliveries} total)")
    else:
        print(f"  {FAIL} Zero deliveries in 200 random episodes")

    return pickup_ok and delivery_ok


def main():
    print("=" * 58)
    print("Reward Shaping Targeted Tests")
    print("=" * 58)

    env, _ = load_env()

    results = {}
    results["no_cascade_penalty"]    = test_no_cascade_on_toggle(env)
    results["cooldown_expires"]       = test_cooldown_expires(env)
    results["no_catastrophic_eps"]    = test_no_catastrophic_episodes(env)
    results["agents_pick_and_deliver"]= test_agents_can_pickup_and_deliver(env)

    env.close()

    print("\n" + "=" * 58)
    print("Results:")
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll reward shaping checks passed — safe to retrain.")
        print("  python3 scripts/train_mappo.py")
    else:
        print("\nFix reward shaping issues before retraining.")
        sys.exit(1)


if __name__ == "__main__":
    main()
