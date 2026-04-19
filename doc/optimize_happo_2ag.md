# HAPPO 2-Agent Optimization: From Barely Beating Random to 7.24 Deliveries/Episode

## Overview

This document records the full optimization journey for 2-agent HAPPO in the `rware-tiny-2ag-v2`
environment — what went wrong in each run, what was diagnosed, what was changed, and why.
The final optimized run achieved a **7× improvement in deliveries** over the original and was
statistically significantly better than random (p=5e-75, Cohen's d=2.04).

---

## Results Summary

| Metric | Original HAPPO (5M) | Optimized HAPPO (8M) | Random Baseline |
|---|---|---|---|
| Mean deliveries/ep | 0.65 | **7.24** | 0.08 |
| Delivery rate | 56.7% | **87.3%** | 8.0% |
| Max deliveries in one ep | 2 | **20** | 1 |
| Positive episode rate | 67.0% | **93.0%** | 7.0% |
| Mean team reward | 2.58 | **30.30** | 0.08 |
| Best eval reward | 5.71 at t=4.63M | **40.82 at t=7.80M** | — |
| Cohen's d (effect size) | 0.68 (medium) | **2.04 (large)** | — |
| Statistically significant | NO | **YES (p=5e-75)** | — |

---

## Part 1: What Was Wrong With the Original Config

### Root Cause 1 — Pickup Reward Too High (`pickup_reward: 0.5`)

The most damaging flaw. With `pickup_reward=0.5`, agents learned to **pick up shelves as a
terminal goal** rather than as a step toward delivery. Evidence:

- Mean pickups per episode: **1.68** (near the max of 2)
- Mean deliveries per episode: **0.65**
- Pickup-to-delivery completion rate: **39%**

Agents discovered that picking up a shelf earned a reliable +0.5 and then either wandered or
dropped the shelf mid-path (earning -0.6 bad_drop_penalty, net = -0.1), which was a small
enough loss to be tolerable. The delivery signal (rware +1 + delivery_bonus +2.0 = +3.0) was
nominally larger, but agents rarely discovered it because they were already locally rewarded
for pickup alone.

### Root Cause 2 — Step Penalty Too Harsh (`step_penalty: -0.005`)

`-0.005 × 500 steps = -2.5 per agent per episode` baseline cost before any delivery. This:

- Caused **33% of all HAPPO episodes to be deeply negative**, swamping the learning signal
- Made the reward landscape dominated by the step penalty rather than delivery signals
- Created a confusing gradient: agents were penalized heavily even when doing the right things

### Root Cause 3 — Training Cut Off Too Early (`total_timesteps: 5M`)

The best eval reward in the original 5M run was at **t=4.63M** — right at the end of training,
still improving. The policy had not converged.

### Root Cause 4 — Network Too Small (`hidden_dim: 128`)

The warehouse task requires agents to plan navigation routes, track shelf/goal positions, and
coordinate with another agent. `hidden_dim=128` provides insufficient capacity for this.

### Root Cause 5 — Rollout Too Short (`n_steps: 512`)

With `max_steps=500` per episode, `n_steps=512` covers only ~1 full episode per rollout.
This means:
- GAE estimates are bootstrapped from an incomplete picture of the return distribution
- High variance in advantage estimates slows learning
- Each actor update uses just 2 minibatches of 128 samples — not enough gradient signal

### Root Cause 6 — Too Conservative Learning Rate (`lr_actor: 0.0002`)

Combined with `n_steps=512` (small rollouts), the effective learning signal per update was
already weak. The conservative lr slowed convergence further. `max_grad_norm=0.5` already
guards against instability, making 0.0002 unnecessarily cautious.

---

## Part 2: The Failed Intermediate Run

Before the optimized run, a report was generated using the **old best_model.pt checkpoint**
(trained under old reward rules) but evaluated in the **new reward environment** (changed
shaping values). This produced catastrophically worse numbers:

- Mean deliveries: **0.023** (worse than random's 0.063)
- Delivery rate: **2.3%** vs random's 6.3%

**This was not a regression in the algorithm** — it was a mismatched evaluation. The old
policy had learned to optimize for `pickup_reward=0.5`; under the new rules where pickup
gives only 0.1, its behavior appeared broken. The new training run had not yet produced a
`best_model.pt`, so the report loaded the stale checkpoint.

**Lesson:** Always regenerate the report *after* the new training run completes and saves
its own `best_model.pt`.

---

## Part 3: All Changes Made and Why

### Reward Shaping (`configs/env_config.yaml`)

| Parameter | Old | New | Reason |
|---|---|---|---|
| `pickup_reward` | 0.5 | **0.1** | Main culprit — taught pickup as terminal goal; agents completed only 39% of pickups into deliveries |
| `delivery_bonus` | 2.0 | **3.0** | Total delivery = rware +1 + 3.0 = **+4.0**; makes delivery unambiguously the dominant reward (40× pickup) |
| `step_penalty` | -0.005 | **-0.001** | Old: -2.5 baseline/ep caused 33% of episodes to be deeply negative. New: -0.5 baseline/ep |
| `bad_drop_penalty` | -0.6 | **-0.8** | Compensates for lower pickup_reward; net of bad pick+drop = 0.1 − 0.8 = **−0.7** (strongly discourages) |
| `linger_penalty` | -0.05 | **-0.02** | Was too harsh when agents approached goal cell; discouraged correct behavior near delivery station |

### HAPPO Hyperparameters (`configs/happo_config.yaml`)

| Parameter | Old | New | Reason |
|---|---|---|---|
| `hidden_dim` | 128 | **256** | Larger capacity for navigation + planning; agents need to track shelf/goal positions and route plans |
| `lr_actor` | 0.0002 | **0.0003** | Was unnecessarily conservative; `max_grad_norm=0.5` already guards stability |
| `n_steps` | 512 | **1000** | 512 ≈ 1 episode → high-variance GAE. 1000 spans **2 full episodes** for accurate return estimates and 8 minibatches/actor/epoch |
| `n_epochs` | 3 | **4** | Larger rollout (1000 vs 512) supports more epochs without overfitting per-actor dataset |
| `clip_epsilon_end` | 0.1 | **0.15** | Less aggressive decay keeps gradient updates productive in the later high-entropy phase |
| `total_timesteps` | 5M | **8M** | Previous best eval at t=4.63M/5M — still improving at cutoff; needed more training |

*Note: `entropy_coef_start=0.05` and `entropy_coef_end=0.01` were already set in the
"v2" config before this optimization pass (they were fixed in a prior iteration to prevent
entropy collapse at t=170k).*

---

## Part 4: Training Dynamics of the Optimized Run

### Phase 1: Exploration (t=0 → ~500k)
- Reward near 0, entropy ~1.2–1.5
- Agents exploring the grid randomly
- No deliveries yet

### Phase 2: Pickup Discovery (~500k → ~1.5M)
- Agents begin reliably picking up shelves
- Training reward rises into positive territory (~3–5)
- Entropy still high, policy not yet decisive

### Phase 3: Delivery Discovery (~1.5M → ~3M)
- Training reward jumps sharply: 5 → 11+
- Eval reward becomes volatile (std=38 at t=2.94M) as policy transitions
- Agents learn the full pick-up → navigate → deliver loop

### Phase 4: Policy Consolidation (~3M → ~8M)
- Training reward stabilizes and climbs steadily: 11 → 30+
- Eval variance drops, reward reliably positive
- Entropy anneals: 1.2 → 0.33 (policy becomes decisive)
- Best eval reward: **40.82 at t=7.8M**
- Final eval: **33.0 ± 19.2, max=70.3**

---

## Part 5: Key Lessons

1. **Pickup reward must be small relative to delivery.** Any intermediate reward large enough
   to create a local optimum will be exploited. The ratio here is 0.1 (pickup) vs 4.0 (delivery)
   = 40×. Agents must be able to discover pickup by accident, not optimize for it directly.

2. **Step penalty baseline kills learning.** A step penalty of `-0.005 × 500 = -2.5` makes
   most episodes negative before any delivery is attempted. Reduce until the baseline cost
   per episode is small relative to a single delivery reward.

3. **Rollout length should span complete episodes.** For a 500-step environment, `n_steps=512`
   is dangerously close to 1 episode. Use at least 2× the episode length (1000) to ensure
   the GAE has enough context to accurately estimate returns.

4. **Train until the curve flattens, not until you run out of budget.** The best checkpoint
   in the original run was at 93% of training time — extending from 5M to 8M steps was
   directly responsible for reaching eval reward 40.8 vs 5.7.

5. **Independent actors need more exploration than shared actors.** HAPPO's per-agent actors
   each see 1/N the data of a MAPPO shared actor. Entropy coefficients must be scaled up
   accordingly to prevent premature collapse.

6. **Always evaluate with the model trained on the same reward structure.** Loading a checkpoint
   trained under old reward rules and evaluating it with new rules produces misleading results
   that look like a regression but aren't.
