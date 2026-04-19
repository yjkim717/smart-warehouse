# HAPPO Implementation — Smart Warehouse

**Algorithm:** Heterogeneous-Agent Proximal Policy Optimization  
**Reference:** Kuba et al., "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning", ICLR 2022  
**Branch:** `feat/happo`  
**File:** `src/algorithms/happo.py`

---

## Table of Contents

1. [What is HAPPO and Why Use It](#1-what-is-happo-and-why-use-it)
2. [HAPPO vs MAPPO — Core Differences](#2-happo-vs-mappo--core-differences)
3. [Architecture](#3-architecture)
4. [The M-Factor: Sequential Update with IS Reweighting](#4-the-m-factor-sequential-update-with-is-reweighting)
5. [Implementation Walkthrough](#5-implementation-walkthrough)
6. [Hyperparameters](#6-hyperparameters)
7. [Training Loop](#7-training-loop)
8. [Checkpointing](#8-checkpointing)
9. [Reused Components](#9-reused-components)
10. [Running HAPPO](#10-running-happo)

---

## 1. What is HAPPO and Why Use It

MAPPO (used in this project's earlier experiments) achieves strong results but relies on **parameter sharing**: every robot uses the same actor network. This works well when all agents are identical — which is true in the rware warehouse setting — but it makes a simplifying assumption that every agent's optimal policy is the same function of its local observation.

HAPPO removes this assumption. Each agent gets its **own independent actor network**, making the algorithm applicable to heterogeneous teams (agents with different sensors, capabilities, or roles). More importantly, HAPPO introduces a theoretically grounded update rule — the **sequential M-factor update** — that provides a **monotonic improvement guarantee** in cooperative multi-agent settings, something vanilla MAPPO lacks.

The practical question this project investigates: does the extra expressiveness (independent actors, principled sequential update) lead to better or faster learning in the rware warehouse task, or does parameter sharing in MAPPO give it a data-efficiency advantage?

---

## 2. HAPPO vs MAPPO — Core Differences

| Aspect | MAPPO | HAPPO |
|--------|-------|-------|
| Actor networks | 1 shared (parameter sharing) | N independent (`nn.ModuleList`) |
| Actor optimizers | 1 `Adam` | N `Adam` — one per actor |
| Update order | All agents updated together per epoch | Sequential random permutation per epoch |
| Advantage used | GAE directly | GAE × M-weights (IS product of prior agents) |
| Critic update timing | Interleaved with actor updates | After all actors per epoch |
| Theoretical guarantee | None for joint update | Monotonic improvement under joint policy |
| Checkpoint key | `"actor"` (single state dict) | `"actors"` (list of state dicts) |
| Supports heterogeneous agents | No | Yes |

Everything else is shared: the same `Critic` network, the same `RolloutBuffer`, the same GAE computation, the same LR/entropy/clip scheduling math, the same obs normalization.

---

## 3. Architecture

### 3.1 Per-Agent Actors

```
actors[0]: Actor(obs_dim=71, action_dim=5, hidden_dim=128, n_layers=2)
actors[1]: Actor(obs_dim=71, action_dim=5, hidden_dim=128, n_layers=2)
...
actors[n-1]: Actor(...)
```

Each actor is a 2-layer MLP with Tanh activations, orthogonal weight initialization, and a categorical output distribution over the 5 discrete actions (Up, Right, Down, Left, Interact). The architecture is **identical** to MAPPO's single actor — the difference is that each agent owns a separate instance with separate weights.

Orthogonal initialization (from `networks.py`):
- Hidden layers: `gain = nn.init.calculate_gain("tanh")` — correct for Tanh activations
- Actor output layer: `gain = 0.01` — near-uniform initial policy, unbiased early exploration
- Bias: zeros everywhere

### 3.2 Shared Centralized Critic

```
critic: Critic(input_dim = n_agents * obs_dim + n_agents, hidden_dim=128, n_layers=2)
```

The critic sees the **global state**: all agents' normalized observations concatenated, plus a one-hot agent ID vector identifying which agent's value is being estimated:

```
global_obs[i] = concat(obs_0, obs_1, ..., obs_{n-1}, one_hot(i))
              = concat(71 × n_agents floats + n_agents floats)
```

For 2 agents: `global_obs_dim = 2×71 + 2 = 144`  
For 4 agents: `global_obs_dim = 4×71 + 4 = 288`

This is the **Centralized Training** part of CTDE — the critic has access to information at training time that actors do not have at execution time. Each agent's value estimate reflects the full team state.

Critic output layer: `gain = 1.0` — initialized to fit the return scale quickly.

### 3.3 Observation Normalization

One **shared `RunningMeanStd`** tracks the running mean and variance of all agents' observations. Since all agents in rware have the same observation space and semantics (same grid encoding, same distances), a shared normalizer is correct and benefits from more data than per-agent normalizers would.

Welford's online algorithm accumulates statistics incrementally:
```
update each rollout: batch mean, batch var, batch count → update running mean/var/count
normalize: (obs - mean) / (sqrt(var) + 1e-8)
```

---

## 4. The M-Factor: Sequential Update with IS Reweighting

This is the algorithm's core theoretical contribution. Here is a precise explanation of what it computes, why, and how it is implemented.

### 4.1 The Problem with Joint Updates

In MAPPO, all agents are updated simultaneously in each minibatch. The PPO objective for agent `i` is:

```
L_i = E[ min(r_i * A_i, clip(r_i, 1-ε, 1+ε) * A_i) ]
where r_i = π_i_new(a_i | o_i) / π_i_old(a_i | o_i)
```

The advantage `A_i` is estimated from the centralized critic **before** any policy updates in this round. When agent `j` updates during the same PPO epoch, `A_i` becomes stale because the joint policy has changed. MAPPO ignores this staleness. HAPPO addresses it.

### 4.2 The Sequential Update Scheme

HAPPO updates agents **one at a time** in each epoch. Before updating agent `i`, it accounts for the fact that agents `1, 2, ..., i-1` have already updated their policies in this epoch. The adjustment is made via importance sampling.

**Algorithm (one epoch):**

```
agent_order = random_permutation([0, 1, ..., n-1])
M_weights   = ones(n_steps)    ← per-step IS product, starts at 1

for each agent_idx in agent_order:

    # The HAPPO advantage for this agent at each step:
    happo_adv[t] = GAE_adv[t, agent_idx] × M_weights[t]

    # PPO update for actors[agent_idx] using happo_adv
    # (standard clipped surrogate over n_steps minibatches)
    ...update actors[agent_idx]...

    # After update: compute IS ratio for this agent over all n_steps
    with no_grad:
        new_lp[t] = log π_agent_idx_NEW(a[t] | o[t])
        old_lp[t] = log π_agent_idx_OLD(a[t] | o[t])   ← from rollout, unchanged
        is_ratio[t] = exp(new_lp[t] - old_lp[t])

    # Accumulate into M_weights for subsequent agents
    M_weights[t] *= is_ratio[t]    ← for all t simultaneously

# After all actors: update shared critic once
```

### 4.3 What M_weights Represents

`M_weights[t]` at the point of updating agent `i` is:

```
M_weights[t] = ∏_{j updated before i} [ π_j_new(a_j_t | o_j_t) / π_j_old(a_j_t | o_j_t) ]
```

This product measures how much the joint policy has shifted (at rollout step `t`) due to all the agents that were updated earlier in this epoch. Multiplying the advantage by this product adjusts the credit signal to reflect the **current** joint policy, not the rollout-time joint policy.

**Intuition:** If agents 0 and 1 updated before agent 2, and agent 0 now assigns higher probability to its action at step `t` (ratio > 1), then the effective advantage for agent 2 at step `t` is scaled up — the team is "leaning in" to that situation more than before, so agent 2's contribution matters more. Conversely, if earlier agents deprioritized that step (ratio < 1), agent 2's advantage is scaled down.

### 4.4 Why This Gives a Monotonic Improvement Guarantee

The HAPPO paper proves that this sequential IS reweighting, combined with PPO-clip, gives a lower bound on the improvement of the joint policy after each epoch. Informally: by accounting for other agents' updates via IS ratios, each agent's PPO step does not inadvertently degrade the overall team performance.

In practice (at this project's scale), M_weights stay close to 1.0 (`m_wt ≈ 0.999–1.001` in training logs) because PPO-clip restricts individual policy changes to ~10–20%. The M-factor's effect is small but principled — it prevents compounding errors from joint updates.

### 4.5 Critical Implementation Details

**M_weights reset per epoch, not per agent:**
```python
for epoch in range(self.n_epochs):
    M_weights = np.ones(self.n_steps)   # ← reset HERE, outside agent loop
    for _agent_idx in agent_order:
        ...
```
If M_weights were not reset between epochs, the IS product would accumulate across multiple epochs and eventually diverge.

**M_weights stay in NumPy, not in PyTorch:**
```python
M_weights = M_weights * is_ratio   # numpy multiply, no gradient
```
M_weights are multiplied into the advantage **before** creating the PyTorch tensor for the minibatch. This ensures no gradient flows through the IS product — the M-factor is a scalar weighting, not a differentiable operation.

**IS ratio uses rollout-time log probs as denominator:**
```python
old_lp_i = self.buffer.log_probs[:, agent_idx]   # collected during rollout, never modified
...
is_ratio = np.exp(new_lp_np - old_lp_i)
```
`old_lp_i` is always the policy's log prob **at rollout collection time**, not after any minibatch updates within the current epoch. This is standard PPO IS — the buffer's log probs are fixed for the duration of the entire update phase, then discarded when `buffer.reset()` is called.

**M_weights computed over all n_steps, not just the last minibatch:**
```python
with torch.no_grad():
    new_lp_all, _ = self.actors[agent_idx].evaluate(obs_t, actions_t)  # obs_t = all n_steps
new_lp_np = new_lp_all.cpu().numpy()
is_ratio   = np.exp(new_lp_np - old_lp_i)   # full n_steps, not a minibatch slice
M_weights  = M_weights * is_ratio
```
After updating `actors[agent_idx]` through multiple minibatches, we do one final `no_grad` forward pass over all `n_steps` to compute the net IS ratio of the fully-updated actor vs. the rollout actor.

---

## 5. Implementation Walkthrough

### 5.1 `__init__`

```python
happo = HAPPO(config, obs_dim=71, action_dim=5, n_agents=2, device="cpu")
```

Key allocations:
```python
self.actors = nn.ModuleList([
    Actor(obs_dim, action_dim, hidden_dim, n_layers).to(device)
    for _ in range(n_agents)
])
self.critic = Critic(global_obs_dim, hidden_dim, n_layers).to(device)

self.actor_optims = [
    torch.optim.Adam(self.actors[i].parameters(), lr=lr_actor)
    for i in range(n_agents)
]
self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

self.buffer = RolloutBuffer(n_steps, n_agents, obs_dim, global_obs_dim)
self.obs_rms = RunningMeanStd((obs_dim,))
```

### 5.2 `select_actions`

Called every step during rollout collection. Loops over each agent's independent actor:

```python
for i in range(self.n_agents):
    obs_i = torch.tensor(norm_obs[i:i+1])   # shape (1, obs_dim)
    a_i, lp_i = self.actors[i].act(obs_i)   # sample from Categorical(logits)
    actions[i]   = a_i.item()
    log_probs[i] = lp_i.item()
```

Return signature is **identical to MAPPO** so the training loop in `train_happo.py` works with zero changes to the rollout collection code.

### 5.3 `update` — Full Flow

```
1. n_epochs iterations:
   a. Random agent permutation (fresh each epoch)
   b. M_weights = ones(n_steps)
   c. For each agent_idx in permuted order:
      i.  Slice obs_i, actions_i, old_lp_i from buffer (shape: n_steps each)
      ii. happo_adv = buffer.advantages[:, agent_idx] * M_weights
      iii. Normalize happo_adv
      iv. Minibatch PPO loop over n_steps (not n_steps*n_agents):
            - ratio = exp(new_lp - old_lp[idx])
            - loss  = -min(ratio*adv, clip(ratio)*adv) - entropy_coef*entropy
            - backward + grad clip + actor_optims[agent_idx].step()
      v.  no_grad full-rollout forward pass → update M_weights
   d. Critic update: minibatch over all n_steps*n_agents samples
      - Clipped value loss (same formula as MAPPO)
      - critic_optim.step()

2. buffer.reset()
3. Return averaged losses
```

### 5.4 `step_schedulers`

Identical math to MAPPO — cosine LR decay, linear entropy annealing, linear clip decay — but iterates over all actor optimizers:

```python
for optim in self.actor_optims:
    for pg in optim.param_groups:
        pg["lr"] = new_lr_actor
```

---

## 6. Hyperparameters

### 2-Agent (`configs/happo_config.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_dim` | 128 | Same as MAPPO proven config |
| `n_layers` | 2 | Same as MAPPO |
| `lr_actor` | 3e-4 | Starting LR, decayed by cosine schedule |
| `lr_critic` | 5e-4 | Critic learns faster |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE trace decay |
| `clip_epsilon_start` | 0.2 | PPO clip range, initial |
| `clip_epsilon_end` | 0.1 | PPO clip range, final (tighter = safer late updates) |
| `entropy_coef_start` | 0.01 | Entropy bonus weight, initial |
| `entropy_coef_end` | 0.003 | Entropy bonus weight, final |
| `value_loss_coef` | 0.5 | Critic loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `lr_decay` | true | Cosine schedule active |
| `lr_min` | 1e-5 | LR floor |
| `n_steps` | 256 | Rollout length before each update |
| `n_epochs` | 4 | PPO update epochs per rollout |
| `minibatch_size` | 128 | Per-agent minibatch (over n_steps dimension) |
| `total_timesteps` | 3,000,000 | Training budget |
| `eval_interval` | 10,000 | Steps between evaluations |
| `eval_episodes` | 20 | Episodes per evaluation |

### 4-Agent (`configs/happo_config_4ag.yaml`)

Changes from 2-agent:
| Parameter | 2-Agent | 4-Agent | Reason |
|-----------|---------|---------|--------|
| `entropy_coef_start` | 0.01 | 0.02 | More exploration needed on larger grid |
| `minibatch_size` | 128 | 256 | Larger absolute batch, same relative to n_steps |
| `total_timesteps` | 3,000,000 | 5,000,000 | Harder coordination problem |
| `checkpoint_dir` | `results/checkpoints_happo` | `results/checkpoints_happo_4ag` | Separate artifacts |
| `log_dir` | `results/logs_happo` | `results/logs_happo_4ag` | Separate artifacts |

All other values are identical — the same proven hyperparameter set from MAPPO scaling.

---

## 7. Training Loop

`scripts/train_happo.py` mirrors `train_mappo.py` with these specific changes:

### Rollout collection — unchanged

```python
actions, log_probs, values, global_obs, norm_obs, _ = happo.select_actions(obs)
next_obs, rews, dones, _ = env.step(actions.tolist())
happo.buffer.insert(norm_obs, global_obs, actions, log_probs, rews, dones, values)
```

The `_` (hidden state) is always `None` — HAPPO uses MLP actors only.

### Logging — adds `m_wt` diagnostic

```
t=    1,024  episodes=2  mean_reward(50)= -4.787  p_loss=-0.0018  ...  m_wt=1.0001  lr=2.56e-04
```

`m_wt` (M-weight mean) reports the average M_weights value at the end of the last update epoch. Values near 1.0 indicate small per-agent policy changes (expected with PPO-clip). Values drifting significantly from 1.0 would signal large policy shifts.

### Evaluation — per-agent greedy actions

```python
for i in range(happo.n_agents):
    obs_i = torch.tensor(norm_obs[i:i+1], device=happo.device)
    logits = happo.actors[i](obs_i)
    actions.append(logits.argmax(dim=-1).item())
```

Each agent uses its own actor for greedy argmax evaluation.

### GIF recording — loads per-agent actor list

```python
actors = [
    Actor(meta["obs_dim"], meta["action_dim"], meta["hidden_dim"], meta["n_layers"])
    for _ in range(n_agents)
]
for i, state in enumerate(ckpt["actors"]):
    actors[i].load_state_dict(state)
```

Output: `results/videos_happo/happo_trained.gif`

---

## 8. Checkpointing

### Checkpoint Format

```python
{
    "actors": [actors[0].state_dict(), actors[1].state_dict(), ...],   # list, n_agents entries
    "critic": critic.state_dict(),
    "actor_optims": [actor_optims[0].state_dict(), ...],               # list, n_agents entries
    "critic_optim": critic_optim.state_dict(),
    "obs_rms": {
        "mean": obs_rms.mean,     # (obs_dim,) array
        "var":  obs_rms.var,      # (obs_dim,) array
        "count": obs_rms.count,   # scalar
    },
    "scheduler_state": {
        "current_timestep": ...,
        "entropy_coef": ...,
        "clip_eps": ...,
    },
    "metadata": {
        "obs_dim": 71, "action_dim": 5, "n_agents": 2,
        "hidden_dim": 128, "n_layers": 2,
        "algo": "happo",    ← identifies checkpoint format
    },
}
```

The `"algo": "happo"` field distinguishes HAPPO checkpoints from MAPPO checkpoints (`"actor"` key vs `"actors"` key).

### Output Paths

| Artifact | 2-Agent | 4-Agent |
|----------|---------|---------|
| Best checkpoint | `results/checkpoints_happo/best_model.pt` | `results/checkpoints_happo_4ag/best_model.pt` |
| Latest checkpoint | `results/checkpoints_happo/latest.pt` | `results/checkpoints_happo_4ag/latest.pt` |
| Eval curve | `results/logs_happo/happo_eval_curve.json` | `results/logs_happo_4ag/happo_eval_curve.json` |
| Training rewards | `results/logs_happo/happo_train_rewards.{json,csv}` | `results/logs_happo_4ag/...` |
| Training plot | `results/plots_happo/happo_training_curve.png` | `results/plots_happo_4ag/...` |
| GIF | `results/videos_happo/happo_trained.gif` | `results/videos_happo_4ag/happo_trained.gif` |

---

## 9. Reused Components

HAPPO deliberately reuses existing code rather than reimplementing it:

| Component | Source | How Used in HAPPO |
|-----------|--------|--------------------|
| `Actor` | `src/algorithms/networks.py` | Instantiated N times in `nn.ModuleList` |
| `Critic` | `src/algorithms/networks.py` | Single shared instance, identical usage |
| `RolloutBuffer` | `src/algorithms/buffer.py` | Same constructor; HAPPO accesses raw arrays (`buffer.obs[:, i, :]`) instead of calling `get_batches()` |
| `RunningMeanStd` | `src/algorithms/mappo.py` (imported) | One shared instance for all agents |
| `RewardTracker` | `src/analytics/__init__.py` | Identical episode logging |
| `WarehouseEnv` | `src/env/warehouse_env.py` | Unchanged — same `reset()`, `step()`, `render()` interface |

The only component **not** reused is `RolloutBuffer.get_batches()`. HAPPO's sequential update requires iterating over agents one at a time, so the per-agent slice `buffer.obs[:, i, :]` is used directly. The buffer itself (storage + `compute_returns`) is reused as-is.

---

## 10. Running HAPPO

### 2-Agent Training

```bash
python scripts/train_happo.py
# or explicitly:
python scripts/train_happo.py \
  --env-config configs/env_config.yaml \
  --happo-config configs/happo_config.yaml
```

### 4-Agent Training

```bash
python scripts/train_happo.py \
  --env-config configs/env_config_4ag.yaml \
  --happo-config configs/happo_config_4ag.yaml
```

### Resume from Checkpoint

```bash
python scripts/train_happo.py \
  --env-config configs/env_config.yaml \
  --happo-config configs/happo_config.yaml \
  --resume results/checkpoints_happo/latest.pt
```

### Generate Comparison Report (after training)

```bash
python scripts/generate_report.py \
  --config configs/env_config.yaml \
  --mappo-config configs/happo_config.yaml \
  --checkpoint results/checkpoints_happo/best_model.pt
```

### Expected Log Output

```
============================================================
HAPPO Training
============================================================
  Env            : rware-tiny-2ag-v2
  Agents         : 2  (each with independent actor)
  Obs dim        : 71
  Action dim     : 5
  Global obs dim : 144
  Device         : cpu
  Total timesteps: 3,000,000
  Rollout length : 256
  LR decay       : True
  Entropy anneal : 0.010 → 0.0030
  Clip eps decay : 0.20 → 0.10
============================================================
  t=    1,024  episodes=2   mean_reward(50)=  X.XXX  p_loss=X.XXXX  v_loss=X.XXXX
               entropy=X.XXXX  ent_coef=X.XXXX  clip_eps=X.XXX  m_wt=X.XXXX  lr=X.XXe-XX  fps=XXXX
  [eval] t=    1,024  reward=X.XXX ± X.XXX  max=X.X  length=XXX
```

The `m_wt` value in the log is the diagnostic for the M-factor. It should stay close to 1.0 throughout training (indicating stable, small per-update policy changes — expected with PPO-clip). A value consistently far from 1.0 would indicate unusually large policy updates and warrants investigation.
