# MAPPO Implementation — Smart Warehouse

**Algorithm:** Multi-Agent Proximal Policy Optimization  
**Reference:** Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2021  
**Files:** `src/algorithms/mappo.py`, `src/algorithms/networks.py`, `src/algorithms/buffer.py`, `scripts/train_mappo.py`

---

## Table of Contents

1. [CTDE Paradigm](#1-ctde-paradigm)
2. [System Overview and Data Flow](#2-system-overview-and-data-flow)
3. [Network Architecture](#3-network-architecture)
4. [Observation Normalization — RunningMeanStd](#4-observation-normalization--runningmeanstd)
5. [Global State Construction](#5-global-state-construction)
6. [Action Selection](#6-action-selection)
7. [Rollout Buffer](#7-rollout-buffer)
8. [GAE Advantage Estimation](#8-gae-advantage-estimation)
9. [PPO Update](#9-ppo-update)
10. [Scheduled Hyperparameters](#10-scheduled-hyperparameters)
11. [GRU Actor (Recurrent Variant)](#11-gru-actor-recurrent-variant)
12. [Training Loop](#12-training-loop)
13. [Evaluation](#13-evaluation)
14. [Checkpointing](#14-checkpointing)
15. [Hyperparameter Reference](#15-hyperparameter-reference)

---

## 1. CTDE Paradigm

MAPPO follows the **Centralized Training, Decentralized Execution (CTDE)** paradigm.

```
Training time                         Execution time
─────────────────────────────────     ──────────────────────────────────
Critic sees GLOBAL state:             Actor sees only LOCAL observation:
  concat(obs_0, obs_1, one_hot(i))      obs_i  shape (71,)
  shape (144,)  for 2 agents

  → can estimate joint value            → must act on partial information
    without communication                 without communication
```

This split solves two problems at once:
- **Training signal quality:** The critic has full information about the team state, so its value estimates are more accurate. Better value estimates → better advantage estimates → cleaner policy gradient signal.
- **Deployment simplicity:** At execution time, each robot only needs its own observation and its own actor. No inter-agent communication is required. The policy can be deployed on independent hardware.

**Parameter sharing:** In this implementation, all agents share a single actor network. Every agent uses the same weights to map its local observation to an action. This is appropriate here because all rware robots are identical — same sensors, same action space, same role. Parameter sharing means the actor sees N times more data than it would with separate per-agent actors, which significantly helps sample efficiency.

---

## 2. System Overview and Data Flow

Below is the complete data flow for one iteration of the training loop.

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING ITERATION                        │
│                                                                  │
│  ┌──────────┐   obs: List[np.ndarray]                           │
│  │   env    │──────────────────────────────────────────────┐    │
│  │ (rware)  │                                              ▼    │
│  └──────────┘                                    ┌──────────────┐│
│       ▲                                          │  normalize   ││
│       │  actions: List[int]                      │  (obs_rms)   ││
│       │                                          └──────┬───────┘│
│       │                                                 │norm_obs│
│       │                              ┌──────────────────┤        │
│       │                              │                  │        │
│       │                      ┌───────▼──────┐   ┌───────▼──────┐│
│       │                      │    Actor     │   │    Critic    ││
│       │                      │  (obs → act) │   │(global_obs→V)││
│       │                      └───────┬──────┘   └───────┬──────┘│
│       │                              │                   │       │
│       │              actions,log_probs                 values    │
│       │                              │                   │       │
│       │                    ┌─────────▼───────────────────▼─────┐ │
│       └────────────────────│        RolloutBuffer              │ │
│                            │  insert(obs,global_obs,actions,   │ │
│  repeat n_steps times      │         log_probs,rews,dones,vals)│ │
│                            └───────────────────────────────────┘ │
│                                            │                      │
│              after n_steps:                ▼                      │
│                            ┌───────────────────────────────────┐ │
│                            │    compute_returns(next_values)   │ │
│                            │    GAE advantages + returns       │ │
│                            └───────────────┬───────────────────┘ │
│                                            │                      │
│              n_epochs × minibatches:       ▼                      │
│                            ┌───────────────────────────────────┐ │
│                            │           PPO update              │ │
│                            │  actor:  clipped surrogate loss   │ │
│                            │  critic: clipped value loss       │ │
│                            └───────────────────────────────────┘ │
│                                                                  │
│              then: step_schedulers(timestep)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Network Architecture

All networks are defined in `src/algorithms/networks.py`.

### 3.1 The `_build_mlp` Helper

All MLP networks in this codebase are built with the same helper:

```python
def _build_mlp(in_dim, hidden_dim, out_dim, n_layers, output_gain=1.0):
    gain = nn.init.calculate_gain("tanh")   # ≈ 1.6733
    layers = []
    d = in_dim
    for _ in range(n_layers):
        linear = nn.Linear(d, hidden_dim)
        nn.init.orthogonal_(linear.weight, gain=gain)   # orthogonal init
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
        d = hidden_dim
    output = nn.Linear(d, out_dim)
    nn.init.orthogonal_(output.weight, gain=output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)
```

**Why Tanh, not ReLU:**  
Tanh bounds activations to (-1, 1). In RL, observations and rewards can have arbitrary scale — bounded activations prevent exploding gradients from large inputs that have not yet been normalized. Tanh also has nonzero gradients everywhere (unlike ReLU's dead zone), which matters for the first few thousand steps before the obs normalizer has good statistics.

**Why orthogonal initialization:**  
PyTorch's default (Kaiming uniform) is designed for ReLU networks. For Tanh, orthogonal initialization is theoretically better because orthogonal weight matrices **preserve the norm** of the input vector as it passes through each layer. This means:
- Activations don't saturate immediately in forward passes
- Gradients don't vanish or explode in backward passes
- Training starts in a well-conditioned regime from step 1

The gain factor for Tanh (`nn.init.calculate_gain("tanh") ≈ 1.6733`) scales the orthogonal matrix so the variance of activations stays near 1.0 throughout the network depth.

**Why `output_gain=0.01` for the actor output and `output_gain=1.0` for the critic output:**  
- Actor output with gain 0.01: all logits start near zero → softmax is nearly uniform → the policy starts with equal ~20% probability for each of the 5 actions. The agent begins exploring all options without bias from random weight initialization.  
- Critic output with gain 1.0: the critic needs to predict returns that could range from roughly -6 to +14 (given the reward shaping). A standard-scale output layer can fit this range quickly. If gain were 0.01, the critic would start predicting near-zero values and take many steps to scale up.

---

### 3.2 Actor (MLP)

```python
class Actor(nn.Module):
    def __init__(self, obs_dim=71, action_dim=5, hidden_dim=128, n_layers=2):
        self.net = _build_mlp(obs_dim, hidden_dim, action_dim, n_layers, output_gain=0.01)
```

**Structure:**
```
obs (71,)
  → Linear(71, 128) + Tanh
  → Linear(128, 128) + Tanh
  → Linear(128, 5)               ← logits, gain=0.01
  → Categorical(logits=logits)   ← action distribution
```

**Parameter count:** 71×128 + 128 + 128×128 + 128 + 128×5 + 5 = **27,013 parameters**

**Key methods:**

`act(obs)` — used during rollout collection:
```python
dist = Categorical(logits=self.net(obs))
action = dist.sample()           # stochastic sampling
log_prob = dist.log_prob(action)
return action, log_prob
```

`evaluate(obs, actions)` — used during the PPO update:
```python
dist = Categorical(logits=self.net(obs))
return dist.log_prob(actions), dist.entropy()
```
The `evaluate` path recomputes log probabilities for **actions that were already taken** during rollout. This is necessary for the importance sampling ratio `π_new / π_old` in the clipped surrogate loss.

`entropy()` returns `H(π) = -∑ π(a) log π(a)` for a Categorical distribution. For a uniform distribution over 5 actions, this is `log(5) ≈ 1.609`. In practice, the starting entropy in logs is `≈1.609`, confirming the near-uniform initialization is working.

---

### 3.3 Critic (MLP)

```python
class Critic(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=128, n_layers=2):
        self.net = _build_mlp(input_dim, hidden_dim, 1, n_layers, output_gain=1.0)
```

**Structure (2-agent case):**
```
global_obs (144,)
  → Linear(144, 128) + Tanh
  → Linear(128, 128) + Tanh
  → Linear(128, 1)               ← scalar value estimate V(s)
  → squeeze(-1)                  ← shape (batch,)
```

**Parameter count (2-agent):** 144×128 + 128 + 128×128 + 128 + 128×1 + 1 = **35,201 parameters**

The critic is called once per step with the global obs for all agents simultaneously:
```python
global_obs_t = torch.tensor(global_obs, device=device)   # shape (n_agents, 144)
values = self.critic(global_obs_t)                        # shape (n_agents,)
```
Each agent gets its own value estimate because its global obs includes a different one-hot agent ID (see Section 5).

---

## 4. Observation Normalization — RunningMeanStd

**File:** `mappo.py`, class `RunningMeanStd`

Raw rware observations are not normalized — values can span different scales depending on which feature is being encoded (grid distances, binary flags, direction encodings). Feeding unnormalized inputs to neural networks causes unstable learning, especially in the early steps when the critic is trying to fit the value function.

`RunningMeanStd` implements **Welford's online algorithm** — it tracks the running mean and variance of all observations seen so far, without storing the full history.

### The Math

**Update rule** (given a new batch of shape `(batch_count, obs_dim)`):

```
delta  = batch_mean - self.mean
total  = self.count + batch_count

new_mean = self.mean + delta × (batch_count / total)

m_a  = self.var × self.count          ← unnormalized variance, old data
m_b  = batch_var × batch_count        ← unnormalized variance, new data
m2   = m_a + m_b + delta² × (self.count × batch_count / total)
new_var = m2 / total
```

The `delta²` cross-term correctly accounts for the difference in means between the old data and the new batch — without it, the running variance would be biased.

**Normalization:**
```python
normalized = (x - self.mean) / (sqrt(self.var) + 1e-8)
```
The `1e-8` prevents division by zero in the early steps when `self.var` is small.

### Usage in MAPPO

```python
self.obs_rms = RunningMeanStd((obs_dim,))   # one shared instance for all agents

# During rollout (update=True):
norm_obs = self._normalize_obs(raw_obs, update=True)
# → obs_rms.update(raw_obs)   ← statistics grow with each rollout step
# → return (raw_obs - mean) / (sqrt(var) + 1e-8)

# During evaluation (update=False):
norm_obs = self._normalize_obs(raw_obs, update=False)
# → skip update, use fixed statistics from training
```

Statistics are accumulated over all agents simultaneously: if `n_agents=2`, each rollout step adds 2 observation vectors to the running statistics. After 256 rollout steps, the normalizer has seen `256 × 2 = 512` samples.

The obs_rms state (mean, var, count) is saved to and loaded from checkpoints, so that resumed training and evaluation use the same normalization that was in effect during the original training run.

---

## 5. Global State Construction

**Method:** `MAPPO.build_global_obs(obs_list)`

The centralized critic needs a representation of the full team state. This is constructed by concatenating all agents' normalized observations into one vector, then appending a one-hot agent ID.

```
For agent i:
global_obs[i] = [obs_0 | obs_1 | ... | obs_{n-1} | 0 ... 1 ... 0]
                 └──── all agents' obs ────┘          └─ one_hot(i) ─┘
```

**Concrete dimensions:**
```
2-agent: global_obs[i] = (71 + 71 + 2,) = (144,)
4-agent: global_obs[i] = (71 + 71 + 71 + 71 + 4,) = (288,)
```

**Why include a one-hot agent ID:**  
Every agent feeds the same concatenated observations to the critic. Without the ID, agent 0 and agent 1 would receive identical inputs and the critic could not produce different value estimates for them. The one-hot flag tells the critic "this value estimate is specifically for agent i's contribution to the joint return."

**Implementation:**
```python
def build_global_obs(self, obs_list: list) -> np.ndarray:
    all_obs = np.concatenate(obs_list)   # (n_agents * obs_dim,)
    global_obs = np.zeros((self.n_agents, self.global_obs_dim), dtype=np.float32)
    for i in range(self.n_agents):
        agent_id = np.zeros(self.n_agents, dtype=np.float32)
        agent_id[i] = 1.0
        global_obs[i] = np.concatenate([all_obs, agent_id])
    return global_obs   # (n_agents, global_obs_dim)
```

---

## 6. Action Selection

**Method:** `MAPPO.select_actions(obs_list)`  
**Called:** Every step during rollout collection  
**Decorated with:** `@torch.no_grad()` — no gradients needed here

```python
@torch.no_grad()
def select_actions(self, obs_list):
    raw_obs  = np.stack(obs_list)               # (n_agents, obs_dim)
    norm_obs = self._normalize_obs(raw_obs)     # normalize + update obs_rms
    obs_t    = torch.tensor(norm_obs, device)   # (n_agents, obs_dim)

    # MLP actor: batch forward pass for all agents simultaneously (parameter sharing)
    actions, log_probs = self.actor.act(obs_t)  # each: (n_agents,)

    # Build global obs and get values from critic
    global_obs   = self.build_global_obs(...)    # (n_agents, global_obs_dim)
    global_obs_t = torch.tensor(global_obs)
    values       = self.critic(global_obs_t)     # (n_agents,)

    return actions.cpu().numpy(), log_probs.cpu().numpy(),
           values.cpu().numpy(), global_obs, norm_obs, hidden_np
```

**Why all agents are batched together:**  
With parameter sharing, the actor is a single network. Passing a batch of shape `(n_agents, obs_dim)` computes all agents' actions in a single forward pass. This is the key efficiency advantage of parameter sharing — the GPU (or CPU) processes all agents simultaneously rather than looping.

**What gets returned and why:**

| Return value | Shape | Purpose |
|---|---|---|
| `actions` | `(n_agents,)` | Passed to `env.step()` |
| `log_probs` | `(n_agents,)` | Stored in buffer as `π_old` for PPO ratio |
| `values` | `(n_agents,)` | Stored in buffer for GAE computation |
| `global_obs` | `(n_agents, global_obs_dim)` | Stored in buffer for critic update |
| `norm_obs` | `(n_agents, obs_dim)` | Stored in buffer for actor update |
| `hidden_np` | `(n_agents, hidden_dim)` or `None` | GRU state, stored for recurrent update |

---

## 7. Rollout Buffer

**File:** `src/algorithms/buffer.py`, class `RolloutBuffer`

The buffer is a fixed-size in-memory store for one complete rollout. It is pre-allocated as NumPy arrays at construction time.

### 7.1 Storage Layout

```python
self.obs         = np.zeros((n_steps, n_agents, obs_dim),        float32)
self.global_obs  = np.zeros((n_steps, n_agents, global_obs_dim), float32)
self.actions     = np.zeros((n_steps, n_agents),                  int64)
self.log_probs   = np.zeros((n_steps, n_agents),                  float32)
self.rewards     = np.zeros((n_steps, n_agents),                  float32)
self.dones       = np.zeros((n_steps, n_agents),                  float32)
self.values      = np.zeros((n_steps, n_agents),                  float32)

self.returns     = np.zeros((n_steps, n_agents),                  float32)   # computed
self.advantages  = np.zeros((n_steps, n_agents),                  float32)   # computed

# Optional GRU hidden state
self.hidden_states = np.zeros((n_steps, n_agents, hidden_dim),    float32)
```

With `n_steps=256` and `n_agents=2`:
- Total transitions per rollout: `256 × 2 = 512`
- Buffer memory (approximate): `512 × (71 + 144 + 1 + 1 + 1 + 1 + 1) × 4 bytes ≈ 440 KB`

### 7.2 Insertion

```python
def insert(self, obs, global_obs, actions, log_probs, rewards, dones, values, hidden=None):
    self.obs[self.pos]        = obs           # (n_agents, obs_dim)
    self.global_obs[self.pos] = global_obs   # (n_agents, global_obs_dim)
    self.actions[self.pos]    = actions      # (n_agents,)
    self.log_probs[self.pos]  = log_probs    # (n_agents,)
    self.rewards[self.pos]    = rewards      # (n_agents,)
    self.dones[self.pos]      = dones        # (n_agents,)
    self.values[self.pos]     = values       # (n_agents,)
    self.pos += 1
```

`self.pos` is a simple integer pointer. After `n_steps` insertions, `self.pos == n_steps` and the buffer is full. The buffer is designed for exactly one rollout — it has no overflow handling and is reset via `reset()` after every update.

### 7.3 Reset

```python
def reset(self):
    self.pos = 0
```

Only the position pointer is reset. The array data is overwritten element by element on the next rollout, so there is no need to zero the arrays. This avoids a `n_steps × n_agents × obs_dim` memset on every iteration.

---

## 8. GAE Advantage Estimation

**Method:** `RolloutBuffer.compute_returns(next_values, gamma, gae_lambda)`  
**Called:** Once after each complete rollout, before the PPO update

GAE (Generalized Advantage Estimation, Schulman et al. 2016) estimates how much better an action was than expected, while balancing bias and variance.

### 8.1 TD Residual

For each timestep `t`, the **TD residual** (one-step advantage estimate) is:

```
δ_t = r_t + γ × V(s_{t+1}) × (1 - done_t) - V(s_t)
```

- `r_t` — shaped reward at step `t`
- `γ × V(s_{t+1})` — discounted estimate of future value
- `(1 - done_t)` — zero mask: if episode ended at `t`, there is no future value
- `V(s_t)` — the critic's prediction of value at step `t`

If `δ_t > 0`, the actual outcome was better than the critic expected → positive advantage → increase probability of that action.  
If `δ_t < 0`, the outcome was worse → negative advantage → decrease probability.

### 8.2 GAE Recursion

Raw TD residuals are high-variance. GAE smooths them by exponentially averaging over future residuals:

```
GAE_t = δ_t + (γλ) × (1 - done_t) × GAE_{t+1}
```

This is computed **backwards** through the rollout (from `t = n_steps-1` down to `t = 0`):

```python
gae = np.zeros(n_agents)                        # GAE_{n_steps} = 0

for t in reversed(range(self.n_steps)):
    next_val = next_values if t == n_steps-1    # V(s_{t+1}) after rollout end
               else self.values[t + 1]          # V(s_{t+1}) from buffer

    mask  = 1.0 - self.dones[t]
    delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
    gae   = delta + gamma * gae_lambda * mask * gae
    self.advantages[t] = gae
```

**The `γλ` tradeoff:**  
- `λ=0`: `GAE_t = δ_t` — pure TD(0). Low variance (uses critic heavily), high bias (depends on critic accuracy).
- `λ=1`: `GAE_t = ∑_{k≥t} γ^{k-t} δ_k` — Monte Carlo. Low bias, high variance.
- `λ=0.95` (used here): balances both. Still relies on the critic but smooths over ~20 steps of future returns.

### 8.3 Returns

```python
self.returns = self.advantages + self.values
```

Returns are the target for the critic: `R_t = GAE_t + V(s_t)`. This is equivalent to the λ-return `V^λ_t` and gives the critic a stable regression target.

### 8.4 Advantage Normalization

After computing all advantages over the full rollout:

```python
adv_flat = self.advantages.reshape(-1)                                 # (n_steps × n_agents,)
self.advantages = (self.advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)
```

**Normalization is done over the entire rollout** (`n_steps × n_agents = 512` values), not per minibatch. This is a deliberate design choice from the MAPPO paper — per-minibatch normalization can shift the sign of advantages within a single epoch, which destabilizes training.

After normalization, advantages have mean ≈ 0 and std ≈ 1. This bounds the policy gradient signal, which helps with the learning rate schedule and prevents large gradient steps from a single lucky (or unlucky) transition.

---

## 9. PPO Update

**Method:** `MAPPO.update()`  
**Called:** Once per rollout, after GAE computation

The update runs `n_epochs` full passes over the rollout data, each time drawing shuffled minibatches of size `minibatch_size`.

### 9.1 Minibatch Iteration

```python
def get_batches(self, minibatch_size, device):
    total = self.n_steps * self.n_agents   # 256 × 2 = 512

    # Flatten (steps, agents) into a single dimension
    flat = {
        "obs":          tensor(self.obs.reshape(total, -1)),        # (512, 71)
        "global_obs":   tensor(self.global_obs.reshape(total, -1)), # (512, 144)
        "actions":      tensor(self.actions.reshape(total)),        # (512,)
        "old_log_probs":tensor(self.log_probs.reshape(total)),      # (512,)
        "returns":      tensor(self.returns.reshape(total)),        # (512,)
        "advantages":   tensor(self.advantages.reshape(total)),     # (512,)
        "old_values":   tensor(self.values.reshape(total)),         # (512,)
    }

    indices = np.random.permutation(total)   # random shuffle
    for start in range(0, total, minibatch_size):
        idx = indices[start : start + minibatch_size]
        yield {k: v[idx] for k, v in flat.items()}
```

With `n_steps=256`, `n_agents=2`, `minibatch_size=128`: each epoch yields `512 / 128 = 4` minibatches.  
With `n_epochs=4`: `4 × 4 = 16` total gradient update steps per rollout.

**Why flatten across both steps and agents:**  
With parameter sharing, the actor treats observations from different agents at different timesteps as exchangeable samples — they are all inputs to the same policy. Flattening maximizes sample diversity within each minibatch and allows the full `n_steps × n_agents` pool to be shuffled together.

### 9.2 Actor Loss — Clipped Surrogate Objective

```python
log_probs, entropy = self.actor.evaluate(batch["obs"], batch["actions"])

ratio = torch.exp(log_probs - batch["old_log_probs"])   # π_new(a|s) / π_old(a|s)
adv   = batch["advantages"]

surr1 = ratio * adv
surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
policy_loss = -torch.min(surr1, surr2).mean()
```

**What the ratio measures:**  
`ratio = π_new(a|s) / π_old(a|s)` is an importance weight. If the new policy assigns higher probability to the action than the old policy did, ratio > 1. If lower, ratio < 1.

**Why the clamp:**  
Without clipping, a single very good or bad experience could cause a large policy update that overshoots the optimum. The clamp `[1-ε, 1+ε]` prevents the ratio from moving too far from 1.0, limiting the size of any single update. With `clip_eps=0.2` (early training), the policy probability is allowed to shift by at most ±20% relative to the old policy before the gradient is cut off.

**Why `min(surr1, surr2)`:**  
The min creates an asymmetry: the surrogate is only clipped when the ratio would otherwise increase the advantage (i.e., when taking the unclipped version would be "too optimistic"). When the ratio decreases the advantage, `surr2 ≥ surr1` so `min` returns the unclipped `surr1` — penalties are not clipped. This prevents the policy from being overly conservative when things go wrong.

**The entropy term:**
```python
actor_loss = policy_loss - self.entropy_coef * entropy_mean
```

Subtracting `entropy_coef × H(π)` encourages the policy to remain stochastic (high entropy = more random). Early in training, this promotes exploration. Later in training, `entropy_coef` is annealed down so the policy can commit to its learned behavior. The entropy of a uniform 5-action distribution is `log(5) ≈ 1.609` — the starting entropy in training logs confirms the near-uniform initialization is correct.

### 9.3 Critic Loss — Clipped Value Function

```python
values     = self.critic(batch["global_obs"])           # new prediction
old_values = batch["old_values"]                        # prediction at rollout time
returns    = batch["returns"]                           # GAE target

values_clipped = old_values + torch.clamp(values - old_values, -self.clip_eps, self.clip_eps)

vf_loss_unclipped = (values - returns).pow(2)
vf_loss_clipped   = (values_clipped - returns).pow(2)
value_loss        = torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
```

**Why clip the value function:**  
During the PPO update, the critic is updated multiple times (`n_epochs × minibatches`) on the same rollout data. Without clipping, the critic could make a large jump in one minibatch that makes the GAE advantage estimates from the same rollout stale. The value clipping restricts each update to stay within `clip_eps` of the rollout-time prediction, preventing runaway critic updates within a single epoch.

The `max(unclipped, clipped)` loss is the **pessimistic bound** — it takes whichever loss is larger. This means the critic can only update when doing so would not violate the clip constraint or when violating the constraint improves the loss. It is the value-function analogue of the actor's `min(surr1, surr2)`.

**The combined update:**
```python
actor_loss  = policy_loss - entropy_coef * entropy_mean
critic_loss = value_loss_coef * value_loss              # scaled by 0.5

self.actor_optim.zero_grad()
actor_loss.backward()
nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm=0.5)
self.actor_optim.step()

self.critic_optim.zero_grad()
critic_loss.backward()
nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm=0.5)
self.critic_optim.step()
```

Actor and critic are updated with **separate backward passes** and separate optimizers. This avoids cross-contamination of gradients between the two networks and allows different learning rates (`lr_actor=3e-4` vs `lr_critic=5e-4`).

**Gradient clipping** (`max_grad_norm=0.5`): applied to both networks. If the L2 norm of all gradients exceeds 0.5, they are scaled down proportionally. This prevents gradient explosions from rare high-advantage transitions.

---

## 10. Scheduled Hyperparameters

**Method:** `MAPPO.step_schedulers(timestep)`  
**Called:** Once per rollout, after the PPO update

Three hyperparameters are scheduled over the course of training.

### 10.1 Cosine Learning Rate Decay

```python
progress       = min(timestep / total_timesteps, 1.0)   # 0.0 → 1.0
cosine_factor  = 0.5 × (1 + cos(π × progress))          # 1.0 → 0.0
new_lr         = lr_min + (lr_base - lr_min) × cosine_factor
```

| Timestep | Progress | Cosine factor | LR (actor) |
|----------|----------|---------------|------------|
| 0 | 0.00 | 1.000 | 3.000e-4 |
| 750k | 0.25 | 0.854 | 2.562e-4 |
| 1.5M | 0.50 | 0.500 | 1.550e-4 |
| 2.25M | 0.75 | 0.146 | 4.380e-5 |
| 3.0M | 1.00 | 0.000 | 1.000e-5 |

**Why cosine, not linear:**  
The cosine schedule is smooth — it starts with a slow decay rate, accelerates through the middle, then slows again near the end. This matches the typical learning curve: early training needs a high LR to make rapid progress, mid-training needs moderate decay to refine the policy, and late training needs a very small LR to stabilize without overshooting the optimum.

**Why this was the most critical improvement:**  
The original MAPPO without LR decay peaked at reward 3.56 at timestep 1.3M, then regressed to -1.57 by end of training. The constant LR kept making large gradient steps after the policy had found a good region of parameter space, pushing it away. With cosine decay, the policy continued improving all the way to the final timestep and reached reward 8.18.

### 10.2 Linear Entropy Annealing

```python
self.entropy_coef = max(
    entropy_coef_end,
    entropy_coef_start - (entropy_coef_start - entropy_coef_end) × progress
)
```

| Timestep | `entropy_coef` |
|----------|----------------|
| 0 | 0.0100 |
| 1.0M | 0.0077 |
| 2.0M | 0.0053 |
| 3.0M | 0.0030 |

**Why anneal entropy:**  
Early training needs exploration — the agent doesn't know which actions lead to deliveries. High entropy keeps the policy stochastic, preventing premature convergence to a suboptimal deterministic policy.  
Late training needs commitment — once the agent has learned the pickup-carry-deliver sequence, continued forced randomness wastes time and prevents the policy from executing cleanly. Annealing the entropy coefficient allows the policy to gradually become more decisive.

### 10.3 Linear Clip Epsilon Decay

```python
self.clip_eps = max(
    clip_eps_end,
    clip_eps_start - (clip_eps_start - clip_eps_end) × progress
)
```

| Timestep | `clip_eps` | Allowed ratio range |
|----------|------------|---------------------|
| 0 | 0.20 | [0.80, 1.20] |
| 1.5M | 0.15 | [0.85, 1.15] |
| 3.0M | 0.10 | [0.90, 1.10] |

**Why decay clip epsilon:**  
Early training allows large policy changes (±20%) so the policy can improve rapidly from a random starting point. Late training restricts changes to ±10% — once a good policy has been found, only small refinement steps are appropriate. Larger steps risk overshooting. This works together with LR decay as double protection against late-training regression.

Note that `clip_eps` is also used for the value clipping term in the critic loss — both tighten together as training progresses.

---

## 11. GRU Actor (Recurrent Variant)

**Class:** `GRUActor` in `networks.py`  
**Enabled by:** `use_gru: true` in config (currently disabled in both configs)

### Architecture

```
obs (71,) → Linear(71,128) + Tanh → encoder output (128,)
          → unsqueeze(0)           → (1, batch, 128)
          → GRU(128→128)           → (1, batch, 128)
          → squeeze(0)             → (batch, 128)
          → Linear(128, 5)         → logits (batch, 5)
```

The GRU maintains a hidden state `h` of shape `(n_layers=1, n_agents, hidden_dim=128)` that persists across timesteps within an episode. This gives the policy a form of memory: it can "remember" what it was doing before, such as which shelf it was approaching or which goal it was navigating toward.

### Hidden State Management

The hidden state is stored in `self._gru_hidden` on the MAPPO object, not inside the network. This allows the training loop to manage it explicitly:

```python
# During rollout:
hidden_np = self._gru_hidden.squeeze(0).cpu().numpy()   # save BEFORE the step
actions, log_probs, self._gru_hidden = self.actor.act(obs_t, self._gru_hidden)

# At episode boundary:
mappo.reset_hidden()   # zero the hidden state for the new episode

# At rollout boundary (not episode boundary):
# hidden state is NOT reset — the rollout may span multiple episodes
```

**Why save hidden state before the step:**  
During the PPO update, each minibatch item needs the hidden state that was active **before** that step's forward pass. If the post-step hidden state were stored, the recurrent context would be one step ahead of the corresponding observation, breaking the sequential dependency.

### GRU PPO Update

```python
# In MAPPO.update(), when use_gru=True:
h = batch["hidden_states"].unsqueeze(0)   # (1, minibatch, hidden_dim)
log_probs, entropy = self.actor.evaluate(batch["obs"], batch["actions"], h)
```

The stored per-step hidden states are used directly as the GRU initial state for each minibatch item. This is an approximation — the hidden state is treated as fixed context rather than being propagated through the full sequence. Full BPTT (backpropagation through time) over the entire rollout would be more accurate but is computationally expensive. The stored-hidden approach is standard in PPO-based recurrent policies and works well in practice.

### Why GRU is Currently Disabled

`use_gru: false` in both configs. The rware observation already contains carrying state, direction, and relative positions of nearby shelves — the key information that temporal memory would provide is already present in the Markovian observation. In experiments, the GRU added training complexity (hidden state resets, buffer overhead) without improving performance. The MLP actor (`Actor`) is used instead.

---

## 12. Training Loop

**File:** `scripts/train_mappo.py`, function `train()`

### 12.1 Outer Loop Structure

```
while timestep < total_timesteps:

    ── rollout collection ──────────────────────── n_steps iterations
    for _ in range(n_steps):
        actions, lp, vals, gobs, nobs, hidden = mappo.select_actions(obs)
        obs2, rews, dones, _  = env.step(actions)
        buffer.insert(nobs, gobs, actions, lp, rews, dones, vals, hidden)
        timestep += 1
        if all(dones): obs = env.reset(); mappo.reset_hidden()

    ── compute GAE ────────────────────────────────────────────────────
    next_values = mappo.get_values(obs)
    buffer.compute_returns(next_values, gamma, gae_lambda)

    ── PPO update ─────────────────────────────────────────────────────
    losses = mappo.update()

    ── step schedulers ────────────────────────────────────────────────
    mappo.step_schedulers(timestep)

    ── logging / eval / save ──────────────────────────────────────────
    if timestep % log_interval < n_steps:    print training stats
    if timestep % eval_interval < n_steps:   evaluate + save best
    if timestep % save_interval < n_steps:   save latest
```

### 12.2 Rollout Collection Details

**Two environments are created:** `env` (for training) and `eval_env` (for evaluation). This ensures that evaluation episodes don't corrupt the training environment's state — goal randomization and episode state are kept separate.

**Episode boundaries within rollouts:**  
Episodes can end mid-rollout. When `all(dones)` is True:
- Episode metrics are logged to `tracker`
- `ep_rewards` and `ep_steps` accumulators are reset
- `env.reset()` is called and `obs` is updated for the next step
- `mappo.reset_hidden()` zeroes the GRU state (no-op if `use_gru=False`)

The rollout continues from the new episode without interrupting the buffer fill. GAE computation handles this correctly via the `mask = 1 - done` term, which zeros out the bootstrapped future value at episode boundaries.

**`next_values` bootstrap:**  
After the rollout loop, `mappo.get_values(obs)` computes the critic's value estimate for the **current** observation (which may be the start of a new episode, or mid-episode if the rollout ended before a done). This is used as `V(s_{T+1})` in the GAE computation for the last step.

### 12.3 Episode Tracking

```python
ep_rewards = np.zeros(n_agents, dtype=np.float64)   # per-agent accumulated rewards
ep_steps   = 0
ep_count   = 0

# On episode end:
team_total = float(ep_rewards.sum())
tracker.episodes.append({
    "episode":             ep_count,
    "n_steps":             ep_steps,
    "agent_total_rewards": ep_rewards.tolist(),
    "team_total_reward":   round(team_total, 4),
    "team_reward_per_step": round(team_total / ep_steps, 6),
    ...
})
recent_ep_rewards.append(team_total)
```

`recent_ep_rewards[-50:]` is used for the rolling mean in the log output — it shows the trend over the last 50 episodes rather than just the last one.

### 12.4 Logging

Every `log_interval` timesteps:

```
t=    1,024  episodes=2  mean_reward(50)= -4.787  p_loss=-0.0018  v_loss=0.0436
             entropy=1.6076  ent_coef=0.0082  clip_eps=0.174  lr=2.56e-04  fps=2063
```

| Field | Meaning |
|-------|---------|
| `mean_reward(50)` | Rolling mean team reward over last 50 episodes |
| `p_loss` | Mean policy loss across last update's minibatches (negative = making progress) |
| `v_loss` | Mean value loss (MSE between critic predictions and GAE returns) |
| `entropy` | Mean policy entropy (starts ~1.609 = uniform, decreases as policy commits) |
| `ent_coef` | Current annealed entropy coefficient |
| `clip_eps` | Current annealed PPO clip epsilon |
| `lr` | Current actor learning rate (post cosine decay) |
| `fps` | Environment steps per second |

### 12.5 Scheduling Correctness

`step_schedulers` is called **after** the PPO update, not before. This means the update at timestep `t` uses the LR and entropy that were current before reaching `t`. The schedule at timestep `t` takes effect for the **next** rollout's update. This is the standard convention and prevents the scheduler from applying a future (lower) LR to an earlier update.

---

## 13. Evaluation

**Function:** `evaluate(env, mappo, n_episodes, max_steps)`  
**Called:** Every `eval_interval=10,000` timesteps during training

```python
def evaluate(env, mappo, n_episodes, max_steps):
    for _ in range(n_episodes):
        obs = env.reset()
        mappo.reset_hidden()

        for step in range(max_steps):
            raw_obs  = np.stack(obs)
            norm_obs = mappo._normalize_obs(raw_obs, update=False)   # frozen stats
            obs_t    = torch.tensor(norm_obs, device=mappo.device)

            with torch.no_grad():
                if mappo.use_gru:
                    logits, mappo._gru_hidden = mappo.actor.forward(obs_t, mappo._gru_hidden)
                else:
                    logits = mappo.actor(obs_t)
                actions = logits.argmax(dim=-1).cpu().numpy()   # greedy

            obs, rews, dones, _ = env.step(actions.tolist())
            total_reward += sum(rews)
            if all(dones): break
```

**Key differences from rollout collection:**

1. **`update=False`** in `_normalize_obs`: the obs_rms statistics are frozen during evaluation. Updating them with eval data would shift the normalization for training in a way not aligned with the training distribution.

2. **Greedy `argmax`** instead of `dist.sample()`: evaluation uses the deterministic best action rather than sampling. This produces less noisy reward estimates for checkpoint comparison, though it can be slightly overoptimistic vs. the stochastic policy used during training.

3. **20 evaluation episodes** (`eval_episodes=20`): 10 episodes was found to be too noisy — lucky or unlucky evaluation runs could save a mediocre checkpoint as "best". 20 episodes provides a more stable signal for checkpoint selection.

4. **Separate `eval_env`**: the eval environment is completely independent from the training environment. Each evaluation episode starts from a fresh `env.reset()`, with randomized goal positions (same randomization as training). This ensures evaluation measures generalization, not memorization of a specific goal layout.

---

## 14. Checkpointing

### Checkpoint Contents

```python
{
    "actor":       actor.state_dict(),         # all network weights and biases
    "critic":      critic.state_dict(),
    "actor_optim": actor_optim.state_dict(),   # Adam moment estimates (m, v, step)
    "critic_optim":critic_optim.state_dict(),
    "obs_rms": {
        "mean":  obs_rms.mean,    # (71,) array
        "var":   obs_rms.var,     # (71,) array
        "count": obs_rms.count,   # scalar, total samples seen
    },
    "scheduler_state": {
        "current_timestep": ...,
        "entropy_coef":     ...,  # current annealed value
        "clip_eps":         ...,  # current annealed value
    },
    "metadata": {
        "obs_dim": 71, "action_dim": 5, "n_agents": 2,
        "hidden_dim": 128, "n_layers": 2, "use_gru": false,
    },
}
```

**Why save optimizer state:**  
Adam maintains per-parameter moment estimates (`m` = exponential moving average of gradients, `v` = moving average of squared gradients) and a step count. These are the "memory" of Adam — without them, resumed training resets to a cold-start optimizer state where early updates are overly cautious (step count = 0) and the moment estimates are zero. Saving and restoring optimizer state ensures the learning dynamics continue smoothly.

**Why save obs_rms:**  
If training is resumed or evaluation is run from a checkpoint, the same normalization statistics must be used as during training. Using different statistics (e.g., a freshly initialized `RunningMeanStd`) would transform observations to a different scale, making the loaded network weights meaningless.

**Why save scheduler_state:**  
The annealed `entropy_coef` and `clip_eps` values at the time of saving must be restored on resume. Otherwise, resumed training would restart with the initial (high) values, undoing the annealing progress.

### Two Checkpoint Files

```
results/checkpoints/best_model.pt   ← saved when eval_mean_reward > best_ever
results/checkpoints/latest.pt       ← saved every save_interval=50,000 steps
```

`best_model.pt` is the checkpoint used for evaluation reports and GIF generation. It captures the policy at its peak performance, not necessarily at the end of training. `latest.pt` allows training to be resumed from an approximate recent state if interrupted.

---

## 15. Hyperparameter Reference

### `configs/mappo_config.yaml`

```yaml
mappo:
  hidden_dim: 128          # width of each hidden layer
  n_layers: 2              # number of hidden layers in both actor and critic
  use_gru: false           # MLP actor (recurrent disabled)

  lr_actor: 0.0003         # Adam starting LR for actor (decays via cosine)
  lr_critic: 0.0005        # Adam starting LR for critic (slightly higher)
  gamma: 0.99              # discount factor
  gae_lambda: 0.95         # GAE trace decay

  clip_epsilon_start: 0.2  # PPO clip range, initial
  clip_epsilon_end:   0.1  # PPO clip range, final (tighter = safer late updates)

  entropy_coef_start: 0.01 # entropy bonus weight, initial
  entropy_coef_end:   0.003# entropy bonus weight, final

  value_loss_coef: 0.5     # scales critic loss before backward
  max_grad_norm: 0.5       # gradient clipping threshold (L2 norm)

  lr_decay: true           # enable cosine LR schedule
  lr_min: 0.00001          # LR floor (1e-5)

  n_steps: 256             # rollout length before each update
  n_epochs: 4              # PPO epochs per rollout
  minibatch_size: 128      # samples per gradient update (512 total / 128 = 4 batches/epoch)

  total_timesteps: 3000000 # training budget
  eval_episodes: 20        # episodes per evaluation (for stable checkpoint selection)
  eval_interval: 10000     # steps between evaluations
  save_interval: 50000     # steps between periodic saves
  log_interval: 2000       # steps between training log lines

  checkpoint_dir: "results/checkpoints"
  log_dir: "results/logs"
```

### Rationale for Non-Obvious Values

| Parameter | Value | Why |
|-----------|-------|-----|
| `lr_critic` > `lr_actor` | 5e-4 vs 3e-4 | Critic needs to fit value function quickly to give good advantage estimates early in training. Actor should update more conservatively. |
| `n_steps = 256` | 256 | Empirically proven stable rollout length. Shorter (128) gives noisier GAE; longer (512) delays updates too much. Restored after failed experiment with 512. |
| `n_epochs = 4` | 4 | Standard PPO. More epochs reuse data efficiently; too many epochs make updates stale and destabilize training. |
| `minibatch_size = 128` | 128 | `n_steps × n_agents / minibatch_size = 256×2/128 = 4` batches per epoch. Proportional to rollout size. |
| `eval_episodes = 20` | 20 | Doubled from 10 after noisy checkpoint selection was observed. Provides a more reliable peak reward estimate. |
| `gae_lambda = 0.95` | 0.95 | Standard value from the PPO paper. Balances bias (λ→0) and variance (λ→1). |
| `value_loss_coef = 0.5` | 0.5 | Standard PPO scaling. Prevents the critic loss from dominating the total loss when gradients are computed separately. |
| `max_grad_norm = 0.5` | 0.5 | Tight clipping — prevents rare high-advantage transitions from causing large gradient spikes that destabilize the shared critic's global-state inputs. |
