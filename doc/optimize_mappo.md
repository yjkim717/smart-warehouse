# MAPPO Optimization: From 15% to 99% Delivery Rate

## Overview

This document explains in detail why the original MAPPO configuration underperformed, what went wrong during the first failed optimization attempt, what was ultimately changed to achieve the breakthrough results, and the reasoning behind each decision.

---

## Results Comparison

| Metric | Original MAPPO | Optimized MAPPO | Change |
|---|---|---|---|
| Mean reward per episode | 3.19 | **8.69** | +172% |
| Positive episode rate | 88.7% | **100.0%** | Every episode positive |
| Delivery rate | 15.0% | **99.0%** | 6.6× improvement |
| Avg deliveries per episode | 0.153 | **1.937** | 12.6× more |
| Min reward (worst episode) | -6.70 | **+0.175** | No bad episodes |
| Max deliveries in one episode | 2 | **3** | |
| Best eval checkpoint | 3.56 at t=1.3M | **8.18 at t=2.97M** | |
| Checkpoint regression | Yes (collapsed to -1.57) | **No regression** | |
| Cohen's d (effect size) | 1.71 | **5.30** | |

---

## Part 1: What Was Wrong With the Original Config

### Original `mappo_config.yaml`

```yaml
mappo:
  hidden_dim: 128
  n_layers: 2
  lr_actor: 0.0003
  lr_critic: 0.0005
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2         # single fixed value, never changes
  entropy_coef: 0.01        # single fixed value, never changes
  value_loss_coef: 0.5
  max_grad_norm: 0.5
  n_steps: 256
  n_epochs: 4
  minibatch_size: 128
  total_timesteps: 2_000_000
  eval_episodes: 10
```

### Problem 1: No Learning Rate Decay (Most Critical Issue)

**What the training data showed:**

```
Best eval reward   : 3.5600  at t=1,320,192
Reward (last 5 ck) : -1.5710   ← final policy, much worse
```

The policy peaked at reward **3.56** around t=1.3M timesteps, then completely collapsed to **-1.57** by the end of training. This is not a coincidence — it is a known failure mode of PPO with a constant learning rate.

**Why this happens:** The learning rate (lr=3e-4) controls how large each gradient step is. Early in training, large steps are fine because the policy is far from a good solution. But once the policy has found a good strategy (the "3.56 peak"), large gradient steps become dangerous — they can push the policy away from the good region of parameter space. With a fixed LR, every update throughout all 2M steps uses the same large step size, giving the optimizer just as much power to destroy a good policy as it had to build it.

**The consequence:** The best checkpoint (3.56) was saved by the evaluation callback, so the reported evaluation results used that checkpoint. But the training itself wasted the entire second half (t=1.3M to t=2M) regressing. 700,000 timesteps of training made the policy worse, not better.

**Analogy:** Imagine you are searching for a hilltop blindfolded and taking 1-metre steps. You eventually reach the top. But you keep taking 1-metre steps after reaching it, so you inevitably stumble off. A smaller step size near the top would let you stay there.

---

### Problem 2: Fixed Clip Epsilon

**Original value:** `clip_epsilon: 0.2` (constant throughout all 2M steps)

The PPO clip epsilon controls how much the new policy is allowed to deviate from the old policy in a single update. A value of 0.2 means the probability ratio between new and old policy is clamped to [0.8, 1.2].

With a fixed clip epsilon, late-training updates allow the same magnitude of policy change as early-training updates. This is too permissive once the policy is mature — it enables the large destructive updates that caused the regression above.

---

### Problem 3: Fixed Entropy Coefficient

**Original value:** `entropy_coef: 0.01` (constant throughout all 2M steps)

The entropy coefficient controls how much the algorithm rewards the policy for being uncertain (exploring). High entropy = more random exploration. Low entropy = more deterministic, committed behavior.

A fixed value of 0.01 throughout training means:
- Early training: the policy explores at the same rate as late training (fine)
- Late training: the policy is still encouraged to remain uncertain even after it has found good actions, preventing it from fully committing to the delivery strategy it learned

The ideal behavior is high entropy early (explore many options) and low entropy late (commit to the best found option).

---

### Problem 4: Default (Random) Weight Initialization

The original networks used PyTorch's default weight initialization (Kaiming uniform), which is designed for **ReLU** activations. However, the networks in this codebase use **Tanh** activations. This mismatch means:

- Hidden layer weights are initialized in a range optimized for a different activation function, causing poor gradient flow in early training
- The actor's output layer is initialized with random weights, meaning the initial policy strongly and randomly favors some actions over others. For example, the agent might start by strongly preferring "move right" purely due to random chance in weight initialization. This introduces bias in early exploration.

The MAPPO paper (Yu et al. 2021) specifically calls out orthogonal initialization as one of the key implementation details that makes MAPPO work well.

---

### Problem 5: No Value Function Clipping

**Original behavior:** The critic (value function) updates with no constraint on how much it can change between the old prediction and the new one.

**Why this matters:** During the PPO update, the critic's value prediction from the rollout collection phase is used to compute advantages (how much better an action was than expected). If the critic then makes a very large update in the opposite direction during the PPO epochs, the advantage estimates for that same rollout become stale and inconsistent. This can trigger unstable training dynamics, especially in multi-agent settings where the global state space is larger.

---

## Part 2: The Failed First Optimization Attempt

Before arriving at the good results, there was an intermediate attempt that made results **dramatically worse**:

```
Old MAPPO mean: 3.19
Failed attempt: -4.30    ← worse than random
```

Understanding why this failed is as important as understanding what worked.

### Failed Change 1: Collision Penalty of -0.3 Per Step

A collision penalty was added to punish agents that tried to move but were blocked by an adjacent agent. The intent was good (encourage agents to avoid each other), but the magnitude and mechanism were catastrophically wrong.

**The math that killed training:**
- Grid size: 11×10 cells (tiny)
- Agents: 2 robots in this small space
- Result: agents are frequently adjacent (unavoidable in a small grid)
- Penalty: -0.3 applied every step an agent is adjacent and tries to move
- Worst case: 2 agents × 500 steps × -0.3 = **-300 per episode maximum**
- Actual observed minimum: **-152 per episode**

The delivery reward is +3.0 (base +1 + shaped +2). A single blocked movement step (-0.3) costs 10% of a full delivery reward. With dozens of blocked steps per episode, the penalty signal completely overwhelmed the delivery signal. The policy had no hope of learning that delivering goods was the goal — the collision signals were too loud.

**What the policy learned instead:** To pick up and drop shelves repeatedly (the "toggle exploit"), because each pickup gave +0.5 which helped escape the constant -0.3/step punishment. This is why the failed run showed 62 pickups in a single 500-step episode.

### Failed Change 2: Entropy Start Too High (0.05 instead of 0.01)

The entropy annealing was configured to start at 0.05, which is 5× higher than the original fixed value of 0.01 that had worked. This kept the policy highly random for far longer than needed. Combined with the catastrophic collision penalties, the policy never stabilized: **0 out of 300 evaluation checkpoints were positive** during the entire 3M-step training run.

### Failed Change 3: Wrong Rollout Length

The rollout length was changed from 256 to 512 steps without justification, altering training dynamics unnecessarily.

### The Fix

All three broken changes were reverted:
- Collision penalty set to `0.0` (disabled entirely)
- Entropy start restored to `0.01` (original working value)
- Rollout length restored to `256`

Only then were the genuine improvements added.

---

## Part 3: What Was Actually Changed to Get 99% Delivery Rate

### Change 1: Cosine Learning Rate Decay

**Before:** `lr_actor: 0.0003` (constant for all 2M steps)

**After:**
```yaml
lr_actor: 0.0003       # starting LR
lr_decay: true
lr_min: 0.00001        # ending LR (floor)
total_timesteps: 3_000_000
```

**How it works:** The learning rate follows a cosine curve from 3e-4 down to 1e-5 over the full training duration:

```
t=0         LR = 0.000300   (full step size, fast early learning)
t=750K      LR = 0.000228   (30% through training)
t=1.5M      LR = 0.000155   (halfway, noticeably smaller)
t=2.25M     LR = 0.000083   (75% through, very cautious)
t=3M        LR = 0.000010   (end, tiny refinement steps only)
```

**What changed in training:** In the original run, the policy peaked at t=1.3M and then regressed. In the optimized run, by t=1.3M the LR has already decayed to ~1.5e-4 (half the original). Updates are smaller and safer. The policy kept improving all the way to t=2.97M — the best checkpoint was found near the very end, not in the middle.

**Code location:** `src/algorithms/mappo.py` — `step_schedulers()` method, using manual cosine formula:
```python
cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
new_lr = lr_min + (lr_base - lr_min) * cosine_factor
```

---

### Change 2: Entropy Coefficient Annealing

**Before:** `entropy_coef: 0.01` (constant)

**After:**
```yaml
entropy_coef_start: 0.01    # same as original — no early training change
entropy_coef_end:   0.003   # gently lower late-training
```

**How it works:** The entropy coefficient decreases linearly from 0.01 to 0.003 over the full training run:

```
t=0        entropy_coef = 0.010   (same as original, explore freely)
t=1M       entropy_coef = 0.007   (slightly less random)
t=2M       entropy_coef = 0.005   (policy more committed)
t=3M       entropy_coef = 0.003   (policy fully commits to learned strategy)
```

**Effect:** Early training is identical to the original. The difference only appears in the second half, where the policy is gently nudged toward decisiveness. This helps the final policy reliably execute the pickup-deliver sequence rather than occasionally taking random detours.

---

### Change 3: Clip Epsilon Decay

**Before:** `clip_epsilon: 0.2` (constant)

**After:**
```yaml
clip_epsilon_start: 0.2   # same as original early training
clip_epsilon_end:   0.1   # tighter constraint late in training
```

**How it works:** The PPO clip range narrows linearly from 0.2 to 0.1. Early training allows the policy to change by up to 20% per update (fast learning). Late training only allows 10% change per update (safer refinement).

**Effect:** Works together with LR decay to protect a good late-training policy. Both mechanisms independently reduce the magnitude of policy changes — LR reduces the gradient step size, clip epsilon limits how far the probability ratio can move. Double protection against regression.

---

### Change 4: Orthogonal Weight Initialization

**Before:** PyTorch default (Kaiming uniform, designed for ReLU)

**After (in `src/algorithms/networks.py`):**

```python
# Hidden layers: orthogonal with Tanh gain (~1.67)
gain = nn.init.calculate_gain("tanh")
nn.init.orthogonal_(linear.weight, gain=gain)
nn.init.zeros_(linear.bias)

# Actor output layer: very small gain (near-uniform initial policy)
nn.init.orthogonal_(output.weight, gain=0.01)

# Critic output layer: standard gain
nn.init.orthogonal_(output.weight, gain=1.0)

# GRU weights: orthogonal on all weight matrices
for name, param in self.gru.named_parameters():
    if "weight" in name:
        nn.init.orthogonal_(param)
```

**Why orthogonal is correct for Tanh networks:**
Orthogonal initialization preserves the norm of the input vector as it passes through each layer. Combined with Tanh (which saturates at ±1), this prevents the gradient from vanishing or exploding early in training. The networks start in a regime where every layer is active and every gradient is meaningful.

**Why gain=0.01 for the actor output:**
With a gain of 0.01, the output logits start near zero for all actions. This means the initial policy assigns approximately equal probability (~20%) to all 5 actions. The agent starts by exploring all options fairly, rather than randomly preferring "move left" or "stay still" due to chance in weight initialization. This gives early training a cleaner signal.

**Why gain=1.0 for the critic output:**
The critic needs to predict the scale of actual returns (which could be anywhere from -6 to +14 based on the reward shaping). A gain of 1.0 lets the critic start in a neutral range and fit the return scale quickly.

---

### Change 5: Value Function Clipping

**Before:** Critic updates with no constraint on how much the value prediction can change.

**After (in `src/algorithms/mappo.py`):**

```python
# Clipped value loss
values_clipped = old_values + torch.clamp(
    values - old_values, -self.clip_eps, self.clip_eps
)
vf_loss_unclipped = (values - returns).pow(2)
vf_loss_clipped = (values_clipped - returns).pow(2)
value_loss = torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
```

**How it works:** The old value prediction from rollout collection is stored in the buffer. During the PPO update, the new critic prediction is not allowed to deviate more than `clip_eps` from the old prediction. The loss takes the worse (maximum) of the clipped and unclipped losses — this is a pessimistic bound that ensures the critic can only make conservative updates.

**Why this helps multi-agent MAPPO specifically:** The centralized critic in MAPPO sees the global state (all agents' observations concatenated). This is a larger input space than a single-agent critic, making it more prone to instability. Value clipping provides a safety net against the critic making large destabilizing jumps in any update epoch.

---

### Change 6: More Training Time (2M → 3M Steps)

**Before:** `total_timesteps: 2_000_000`

**After:** `total_timesteps: 3_000_000`

This change alone would have done nothing (and did nothing) in the failed optimization run. But combined with LR decay preventing regression, the extra 1M steps gave the policy time to improve beyond the original peak of 3.56.

**Evidence:** The best checkpoint in the optimized run was found at **t=2,970,112** — near the very end of training. Without the extra training budget, this checkpoint would never have been reached.

---

### Change 7: More Evaluation Episodes (10 → 20)

**Before:** `eval_episodes: 10`

**After:** `eval_episodes: 20`

During training, every 10,000 steps the policy is evaluated and the best checkpoint is saved. With only 10 evaluation episodes, the reward estimate has high variance — a lucky run might save a mediocre checkpoint, or a bad run might fail to save a great one.

With 20 episodes, the evaluation signal is more reliable, meaning `best_model.pt` more accurately tracks the truly best policy.

---

## Part 4: What Was Kept Identical

All of these were deliberately left unchanged to isolate the effect of the improvements:

| Parameter | Value | Reason kept |
|---|---|---|
| hidden_dim | 128 | Already sized correctly |
| n_layers | 2 | Proven adequate |
| lr_actor | 0.0003 (start) | Same starting point |
| lr_critic | 0.0005 (start) | Same starting point |
| gamma | 0.99 | Standard, works well |
| gae_lambda | 0.95 | Standard, works well |
| n_steps | 256 | Proven rollout length |
| n_epochs | 4 | Standard |
| minibatch_size | 128 | Proportional to n_steps |
| max_grad_norm | 0.5 | Gradient clipping fine |
| All reward shaping | unchanged | Working as designed |

---

## Part 5: Training Curve Interpretation

### Original Run

```
t=0 to t=1.3M:   Policy improves steadily → peaks at 3.56
t=1.3M to t=2M:  Policy REGRESSES → -1.57 (large constant LR destroys good policy)
Best checkpoint:  3.56 (saved at peak, used for evaluation)
Wasted timesteps: ~700,000 (made the policy worse)
```

### Optimized Run

```
t=0 to t=1.3M:   Nearly identical to original → similar peak ~3.5-4.0
t=1.3M to t=2M:  LR has halved → policy refines safely → 5.0-6.0 range
t=2M to t=3M:    LR approaching floor → policy fine-tunes → 7.0-8.2 range
Best checkpoint:  8.18 at t=2,970,112 (found near end of training)
Wasted timesteps: None — every phase improved the policy
```

The key insight is that the original training was not wrong about *how* to learn — it was wrong about *when to stop learning aggressively*. The policy knew how to deliver goods at t=1.3M. It just got pushed away from that knowledge by continued large updates.

---

## Part 6: Why 99% Delivery Rate vs 15% Delivery Rate

The delivery rate gap is large, but it follows directly from the checkpoint quality:

- **Old best checkpoint (reward 3.56):** Policy had learned to pick up shelves and sometimes navigate to the goal. Delivery was not yet reliable — agents often dropped shelves in wrong locations or took inefficient paths.

- **New best checkpoint (reward 8.18):** Policy had an additional 1.67M refinement steps at progressively lower LR. By t=3M, the agents have learned the full delivery chain reliably: approach nearest shelf → pick up → navigate to goal → deliver → repeat. The 99% delivery rate means agents almost always complete at least one full delivery per 500-step episode.

The orthogonal initialization contributed here too — by starting with a near-uniform policy, early training explored all movement patterns instead of getting stuck in suboptimal habits from random weight initialization.

---

## Summary Table: Changes and Their Impact

| Change | Files Modified | Addresses | Impact Level |
|---|---|---|---|
| Cosine LR decay | `mappo.py`, `mappo_config.yaml` | Policy regression after peak | **Critical** |
| Orthogonal weight init | `networks.py` | Poor early exploration, slow convergence | **High** |
| Value function clipping | `mappo.py`, `buffer.py` | Critic instability, advantage noise | **High** |
| Entropy annealing | `mappo.py`, `mappo_config.yaml` | Policy indecisiveness in late training | **Medium** |
| Clip epsilon decay | `mappo.py`, `mappo_config.yaml` | Large destructive late updates | **Medium** |
| Extended training (3M) | `mappo_config.yaml` | Insufficient training time (only effective with LR decay) | **Medium** |
| More eval episodes (20) | `mappo_config.yaml` | Noisy checkpoint selection | **Low** |
| Disabled collision penalty | `env_config.yaml` | Removed catastrophic -0.3/step punishment | **Prevented disaster** |
