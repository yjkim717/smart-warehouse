# Scaling Analysis: Reward Hacking in 4-Agent MAPPO

## Overview

When scaling the optimized MAPPO policy from 2 agents (tiny grid) to 4 agents (small grid), the policy achieved high shaped reward but near-zero actual deliveries. This document explains the root cause, presents the evidence, and describes the fix.

---

## The Problem

| Metric | 2-Agent MAPPO | 4-Agent MAPPO | Random (4-agent) |
|---|---|---|---|
| Mean team reward | 8.69 | 9.74 | 0.05 |
| Positive episode rate | 100.0% | 99.0% | 5.3% |
| **Delivery rate** | **99.0%** | **4.7%** | **5.3%** |
| **Mean deliveries/ep** | **1.937** | **0.053** | **0.053** |
| Mean pickups/ep | 2.007 | 4.013 | 4.000 |
| Per-agent mean reward | ~4.34 | ~2.44 | ~0.01 |
| Cohen's d vs random | 5.30 | 4.46 | -- |

The 4-agent policy scores higher team reward than the 2-agent policy (9.74 vs 8.69) while delivering almost nothing (4.7% vs 99.0%). The delivery rate is effectively indistinguishable from the random baseline (5.3%).

---

## Root Cause: Reward Shaping Does Not Scale

### The Reward Shaping Structure

The environment applies dense intermediate rewards to guide learning:

```
Per agent, per step:
  +0.5   when picking up a shelf
  +0.05  per step while carrying and approaching goal
  +0.05  per step while approaching nearest shelf (when empty-handed)
  +3.0   on delivery (rware base +1 + delivery_bonus +2)
  -0.005 per step (time penalty)
  -0.6   for dropping shelf away from goal
  -0.05  per step when lingering on goal while carrying
```

### Why It Works at 2 Agents

With 2 agents, the maximum intermediate reward per episode (without any delivery) is:

```
2 agents × (0.5 pickup + 0.05 × ~200 approach steps) = 2 × 10.5 ≈ 21
Minus time penalty: 2 × 500 × 0.005 = 5
Net intermediate maximum ≈ 16
```

But each delivery adds +3.0, and with a small 11x10 grid agents frequently reach goals within ~50-100 steps. Two successful deliveries add +6.0, making the delivery path significantly more rewarding than the carry-only path. The optimal policy is: **pick up, deliver, repeat.**

### Why It Breaks at 4 Agents

With 4 agents, the intermediate reward ceiling doubles:

```
4 agents × (0.5 pickup + 0.05 × ~150 approach steps) = 4 × 8.0 ≈ 32
Minus time penalty: 4 × 500 × 0.005 = 10
Net intermediate maximum ≈ 22
```

The delivery bonus remains +3.0 per delivery. But on the larger 20x10 grid, deliveries are harder to complete (longer navigation, more agents blocking each other). The policy discovered that it can earn ~10 shaped reward per episode simply by:

1. Each agent picks up a shelf (+0.5 each = +2.0 total)
2. Each agent walks toward a goal while carrying (+0.05/step each)
3. Never actually drop or deliver

This is a **local optimum** in the reward landscape. The carry-toward-goal signal provides a constant positive gradient that the policy follows indefinitely, without needing to solve the harder coordination problem of actually completing the delivery (navigate to exact goal cell, execute interact action).

### This Is Called Reward Hacking

Reward hacking occurs when an agent optimizes a proxy reward (the shaped signal) without optimizing the true objective (task completion). The shaped reward was designed to guide agents toward deliveries, but at 4-agent scale the intermediate signals are large enough to sustain a high-reward policy that never completes the actual task.

---

## Evidence That the Algorithm Is Not at Fault

The MAPPO implementation is correct. All evidence points to reward design, not algorithm failure:

| Evidence | What It Shows |
|---|---|
| 440/500 positive training evals (88%) | Training converged stably across 5M timesteps |
| Best eval reward 10.90 at t=4.46M | Policy continued improving, no regression |
| End entropy 0.079 | Policy committed to deliberate, non-random behavior |
| 99% positive episode rate at eval | Agents are navigating purposefully |
| Mean pickups 4.01 (= n_agents) | Every agent picks up exactly one shelf per episode |
| Mean deliveries 0.053 ≈ random 0.053 | Zero learned delivery behavior despite 5M steps |

The algorithm maximized the reward signal it was given. The signal was wrong.

Switching to a different algorithm (HAPPO, QMIX, etc.) would not fix this problem because any reward-maximizing algorithm would find the same local optimum in this reward landscape.

---

## The Fix: Reward Rebalancing

### Principle

Reduce intermediate reward magnitudes so the delivery bonus (+3.0) is the dominant signal at any agent count. The intermediate signals should be just large enough to guide exploration, not large enough to sustain a non-delivering policy.

### Specific Changes in `configs/env_config_4ag.yaml`

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `pickup_reward` | 0.5 | **0.1** | A pickup is worth 1/30 of a delivery, not 1/6 |
| `carry_toward_goal` | 0.05 | **0.01** | 150 carry steps = +1.5, well below one delivery (+3.0) |
| `move_toward_shelf` | 0.05 | **0.01** | Symmetric reduction for consistency |

Additionally, `step_penalty` was reduced from -0.005 to **-0.002** after the first retrain revealed a second scaling issue: the per-episode penalty scaled to -10.0 with 4 agents (4 × 500 × 0.005), overwhelming even the rebalanced positive rewards. The new value gives -4.0/episode (4 × 500 × 0.002), comparable to the 2-agent penalty of -5.0 (2 × 500 × 0.005).

| Parameter | Before | After | Rationale |
|---|---|---|---|
| `step_penalty` | -0.005 | **-0.002** | 4×500×0.002 = -4.0/ep, comparable to 2-agent's -5.0 |

All other shaping values remain unchanged:
- `delivery_bonus: 2.0` (total +3.0 per delivery)
- `bad_drop_penalty: -0.6`
- `linger_penalty: -0.05`

### Expected Effect on Reward Budget

**Before (broken):**
```
4 agents × (0.5 pickup + 0.05 × 150 carry steps) = 4 × 8.0 = 32 intermediate
vs. +3.0 per delivery
→ Intermediates dominate. No incentive to deliver.
```

**After (fixed):**
```
4 agents × (0.1 pickup + 0.01 × 150 carry steps) = 4 × 1.6 = 6.4 intermediate
Time penalty: 4 × 500 × 0.002 = 4.0
Net intermediate: 6.4 - 4.0 = +2.4
vs. +3.0 per delivery
→ A single delivery (+3.0) already exceeds all net intermediates.
  Two deliveries (+6.0) make the policy clearly profitable.
→ The optimal policy must deliver to maximize reward.
```

### Why Not Change the 2-Agent Config?

The 2-agent config already achieves 99% delivery rate with the original shaping values. The reward hierarchy is correct at that scale. Changing it would risk breaking a working system. The fix is applied only to `env_config_4ag.yaml`.

---

## Broader Lesson: Reward Shaping and Agent Count

This scaling failure reveals a general principle in multi-agent RL:

**Dense reward shaping must be designed relative to the number of agents.** When intermediate per-agent rewards are fixed and the number of agents increases, the total intermediate signal grows linearly while the task-completion signal (delivery bonus) stays constant per event. At some agent count, the intermediate signals cross the threshold where they can sustain a non-completing policy.

A more robust approach would be to scale intermediate rewards inversely with agent count:

```
carry_toward_goal_effective = carry_toward_goal_base / n_agents
```

This ensures the total intermediate budget remains constant regardless of team size. However, for this project we apply a simpler manual rebalancing specific to the 4-agent case.

---

## Retrain Commands

After applying the config fix:

```bash
# Train 4-agent MAPPO with rebalanced rewards
python3 scripts/train_mappo.py \
  --env-config configs/env_config_4ag.yaml \
  --mappo-config configs/mappo_config_4ag.yaml

# Generate comparison report
python3 scripts/generate_report.py \
  --config configs/env_config_4ag.yaml \
  --mappo-config configs/mappo_config_4ag.yaml
```

---

## Summary

| Aspect | Finding |
|---|---|
| Problem | 4-agent MAPPO has 4.7% delivery rate despite 9.74 mean reward |
| Root cause | Reward hacking: intermediate shaping signals dominate task-completion signal at 4-agent scale |
| Algorithm fault? | No — MAPPO converged correctly and maximized the given reward |
| Fix | Reduce `pickup_reward` (0.5→0.1), `carry_toward_goal` (0.05→0.01), `move_toward_shelf` (0.05→0.01), `step_penalty` (-0.005→-0.002) |
| Expected outcome | Delivery bonus becomes dominant signal; policy must deliver to maximize reward |
| Broader lesson | Dense reward shaping magnitude must be considered relative to agent count |
