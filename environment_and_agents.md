# Environment and Agent Setup — Smart Warehouse

**Simulator:** rware (Robotic Warehouse) v2  
**Wrapper:** `src/env/warehouse_env.py`  
**Task:** Cooperative shelf retrieval in an Amazon Kiva-style warehouse grid

---

## Table of Contents

1. [Task Description](#1-task-description)
2. [Grid World](#2-grid-world)
3. [rware: The Underlying Simulator](#3-rware-the-underlying-simulator)
4. [WarehouseEnv Wrapper](#4-warehouseenv-wrapper)
5. [Observations](#5-observations)
6. [Actions](#6-actions)
7. [Reward Structure](#7-reward-structure)
8. [Reward Shaping — Design and Parameters](#8-reward-shaping--design-and-parameters)
9. [Episode Mechanics](#9-episode-mechanics)
10. [Rendering](#10-rendering)
11. [Configuration Files](#11-configuration-files)
12. [Agent Setup in Algorithms](#12-agent-setup-in-algorithms)
13. [Quick Start](#13-quick-start)

---

## 1. Task Description

The task is **cooperative Multi-Agent Path Finding with delivery** (MAPF+D), inspired by Amazon Kiva warehouse robots.

A fleet of autonomous robots operates inside a warehouse grid. Shelves are arranged in rows. Robots must:

1. Navigate to a shelf (avoiding other robots and shelf obstacles)
2. Pick up the shelf with the `Interact` action
3. Carry the shelf to a **packing station** (delivery goal)
4. Deliver the shelf at the goal with `Interact`
5. Repeat

The robots receive a reward for every successful delivery. The objective is to **maximize total deliveries per episode** across the entire team.

**Key challenge:** The environment is only partially observable. Each robot sees a local window of the grid around itself — it cannot see the full warehouse. Robots must implicitly coordinate to avoid blocking each other and maximize throughput without any direct communication.

---

## 2. Grid World

Two grid sizes are used, selected via the environment config:

### Tiny (2-Agent)
```
Environment ID : rware-tiny-2ag-v2
Grid size      : ~11 × 10 cells
Agents         : 2 robots
Shelves        : arranged in rows (most cells)
Packing stations: 2 yellow P cells (randomized each episode)
```

```
Visual layout (approximate):
┌──────────────────────────┐
│ S  S  S  S  S  S  S  S  │   S = shelf (blue)
│ S  S  S  S  S  S  S  S  │   P = packing station (yellow)
│ .  .  .  .  .  .  .  .  │   . = empty aisle
│ S  S  S  S  S  S  S  S  │   ↑ = robot (red/green)
│ P  .  .  .  .  .  .  P  │
└──────────────────────────┘
```

### Small (4-Agent)
```
Environment ID : rware-small-4ag-v2
Grid size      : ~20 × 10 cells
Agents         : 4 robots
Packing stations: 4 yellow P cells (randomized each episode)
```

The small grid is approximately twice the area of the tiny grid, with proportionally more shelf rows and wider aisles for 4 robots to navigate without constant collisions.

---

## 3. rware: The Underlying Simulator

The project uses [rware](https://github.com/semitable/robotic-warehouse) (Robotic Warehouse), a Gymnasium-compatible multi-agent environment.

### Installation

```bash
pip install rware gymnasium
```

### Registration

When `import rware` is executed, all rware variants are registered with Gymnasium:

```python
gym.make("rware-tiny-2ag-v2")   # 2-agent tiny grid
gym.make("rware-small-4ag-v2")  # 4-agent small grid
```

### rware Internal State

The `env.unwrapped` object exposes:
- `env.unwrapped.grid` — 3D grid array, shape `(layers, rows, cols)`
- `env.unwrapped.agents` — list of Agent objects with `.x`, `.y`, `.dir`, `.carrying_shelf`
- `env.unwrapped.shelfs` — list of Shelf objects with `.x`, `.y`
- `env.unwrapped.goals` — list of `(col, row)` tuples for packing stations

### rware Reward

rware gives a raw reward of **+1.0** per delivery (when a robot executes Interact at a packing station while carrying a shelf). All other steps return 0.0. No penalty, no shaping.

The wrapper adds dense shaping on top of this raw reward.

---

## 4. WarehouseEnv Wrapper

`src/env/warehouse_env.py` is a thin wrapper around rware that:

1. Standardizes the multi-agent interface for training algorithms
2. Manages reward shaping state across steps
3. Provides a headless matplotlib renderer (rware-v2's built-in RGB renderer is broken)
4. Enforces episode step limits
5. Optionally randomizes packing station positions each episode

### Interface

```python
from src.env.warehouse_env import WarehouseEnv
import yaml

with open("configs/env_config.yaml") as f:
    config = yaml.safe_load(f)

env = WarehouseEnv(config)

# Reset — returns list of per-agent observations
obs = env.reset()                          # List[np.ndarray], length = n_agents

# Step — takes list of integer actions
obs, rews, dones, info = env.step(actions) # actions: List[int]
# obs   : List[np.ndarray]   — updated observations
# rews  : List[float]        — per-agent rewards (shaped or raw)
# dones : List[bool]         — per-agent done flags
# info  : dict

# Properties
env.n_agents     # number of robots (int)
env.obs_dim      # observation vector length (int)
env.action_dim   # 5 (Up, Right, Down, Left, Interact)

# Render
frame = env.render()   # np.ndarray, shape (H, W, 3), dtype uint8
```

### Constructor Parameters (from config)

```yaml
env:
  name: "rware-tiny-2ag-v2"   # which rware preset to load
  n_agents: 2                  # must match the preset
  max_steps: 500               # episode step limit (enforced by wrapper)
  randomize_goals: true        # shuffle packing station positions each reset
  reward_shaping:
    enabled: true
    ...
  record:
    fps: 8
    output_dir: "results/videos"
```

---

## 5. Observations

### Structure

Each agent receives its own **local observation vector** of shape `(obs_dim,)`, where `obs_dim = 71` for both the tiny-2ag and small-4ag environments.

The observation is computed by rware and encodes:
- **Relative positions and states of nearby shelves** — which cells have shelves, whether they are carried
- **Relative positions of nearby robots** — their location and carrying state
- **Agent's own state** — current direction, whether carrying a shelf
- **Distance/direction to nearest goals** — relative to current position

The exact encoding is rware's internal sensor model (a flattened local grid view with feature channels per cell). The key property: **each agent only sees a limited window around itself**, not the full grid. This makes the problem partially observable.

### Observation Flow in the Wrapper

```python
def reset(self) -> List[np.ndarray]:
    obs, _ = self._env.reset()          # raw obs from rware
    return self._unpack_obs(obs)         # list of per-agent np.float32 arrays

def _unpack_obs(self, obs) -> List[np.ndarray]:
    if isinstance(obs, (list, tuple)):
        return [np.array(o, dtype=np.float32) for o in obs]
    return [np.array(obs[i], dtype=np.float32) for i in range(self._n_agents)]
```

### Observation Normalization (in algorithms)

Raw observations are normalized by the algorithm's `RunningMeanStd` before being passed to the actor networks:

```python
norm_obs = (obs - running_mean) / (sqrt(running_var) + 1e-8)
```

Statistics are updated online during rollout collection (`update=True`) and frozen during evaluation (`update=False`).

---

## 6. Actions

Each agent selects one discrete action per step from a space of **5 actions**:

| Index | Action | Description |
|-------|--------|-------------|
| 0 | Up | Move one cell upward |
| 1 | Right | Move one cell rightward |
| 2 | Down | Move one cell downward |
| 3 | Left | Move one cell leftward |
| 4 | Interact | Pick up shelf (if empty-handed on shelf cell) OR deliver (if carrying on goal cell) |

**Movement rules:**
- Agents cannot move through shelves they are not carrying
- If an agent tries to move into an occupied cell, the move is blocked (agent stays put)
- Carrying a shelf does not block the destination cell for other robots

**Interact rules:**
- Empty-handed agent on a shelf cell → picks up the shelf, becomes a carrying agent
- Carrying agent on a goal (packing station) cell → delivers the shelf (+1 rware reward), becomes empty-handed
- Interact in any other situation → no effect

---

## 7. Reward Structure

### Base Reward (from rware)

```
+1.0  per delivery (carrying agent executes Interact at a packing station)
 0.0  all other steps
```

This is a **very sparse** reward. Without shaping, an agent taking random actions in a 500-step episode has roughly a 5% chance of making even one delivery.

### Shaped Reward (from WarehouseEnv wrapper)

When `reward_shaping.enabled: true`, the wrapper adds dense intermediate rewards on top of the base rware reward. The purpose is to provide a learning signal strong enough to guide agents toward the full delivery sequence, not just the final reward.

Shaped reward per agent per step:

```
base_reward     =  0.0 or +1.0  (from rware)
+delivery_bonus                  on delivery (adds to the +1.0 above)
+pickup_reward                   when agent picks up a shelf
+carry_toward_goal               when carrying and moving closer to goal
-carry_toward_goal × 0.5        when carrying and moving away from goal
+move_toward_shelf               when empty-handed and moving closer to nearest shelf
-move_toward_shelf × 0.5        when empty-handed and moving away from nearest shelf
-bad_drop_penalty                when dropping a shelf NOT at a packing station
-linger_penalty                  per step when carrying on top of a goal (stuck without delivering)
+step_penalty                    per step (negative — small time cost to encourage speed)
```

Distance signals use **Manhattan distance** to the nearest relevant cell (goal for carrying agents, shelf for empty agents).

---

## 8. Reward Shaping — Design and Parameters

### 2-Agent Config (`configs/env_config.yaml`)

```yaml
reward_shaping:
  enabled: true
  pickup_reward: 0.5          # +0.5 on pickup
  delivery_bonus: 2.0         # total delivery = rware +1 + 2.0 = +3.0
  step_penalty: -0.005        # per step
  carry_toward_goal: 0.05     # per step while approaching goal carrying
  move_toward_shelf: 0.05     # per step while approaching shelf empty-handed
  bad_drop_penalty: -0.6      # dropping shelf away from goal
  linger_penalty: -0.05       # per step sitting on goal while carrying without delivering
  collision_penalty: 0.0      # disabled
```

### 4-Agent Config (`configs/env_config_4ag.yaml`)

Intermediate signals are **reduced** to prevent reward hacking at 4-agent scale:

```yaml
reward_shaping:
  enabled: true
  pickup_reward: 0.1          # was 0.5 — reduced to prevent pickup-farming
  delivery_bonus: 2.0         # unchanged — delivery = +3.0
  step_penalty: -0.002        # was -0.005 — 4×500×0.002=-4.0, comparable to 2-agent's -5.0
  carry_toward_goal: 0.01     # was 0.05 — reduced 5× so delivery signal dominates
  move_toward_shelf: 0.01     # was 0.05 — symmetric reduction
  bad_drop_penalty: -0.6      # unchanged
  linger_penalty: -0.05       # unchanged
  collision_penalty: 0.0      # disabled
```

### Why the 4-Agent Values Are Different

With fixed intermediate rewards, total shaped reward scales linearly with agent count while the delivery bonus stays constant per delivery event. At 4-agent scale, the original shaping values allowed agents to accumulate high reward by picking up shelves and walking toward goals indefinitely — without ever completing a delivery. This is **reward hacking**.

The fix: reduce intermediate signals so that one delivery (+3.0) exceeds the maximum total intermediate reward an episode. See `scaling_analysis.md` for the full root-cause analysis.

### Reward Hacking Prevention Details

**Drop cooldown:** After an agent drops a shelf away from the goal (triggering `bad_drop_penalty`), pickup reward is suppressed for 5 steps. This prevents the agent from immediately re-picking the same shelf to farm the pickup bonus.

```python
self._drop_cooldown[i] = 5   # set on bad drop
# each step: if cooldown > 0, pickup bonus = 0; cooldown -= 1
```

**Asymmetric distance shaping:** Approaching a goal/shelf gives the full signal; retreating only incurs half the penalty. This creates a strong gradient toward the goal without catastrophically punishing occasional backward steps.

```python
if dist < prev:
    shaped[i] += self._carry_toward_goal          # +full
elif dist > prev:
    shaped[i] -= self._carry_toward_goal * 0.5   # -half
```

**Linger penalty:** If an agent is already standing on a goal cell while carrying a shelf (distance = 0), it receives a penalty each step until it executes Interact to deliver. This prevents an agent from standing on the goal indefinitely collecting `carry_toward_goal` reward without finishing the delivery.

**Collision penalty disabled:** Set to 0.0 in both configs. A per-step collision penalty of -0.3 was tested and catastrophically failed: in an 11×10 grid with 2 robots, agents are frequently adjacent to each other, and the penalty signal completely overwhelmed the delivery signal (see `optimize_mappo.md`).

---

## 9. Episode Mechanics

### Step Limit

Each episode runs for at most `max_steps = 500` steps. The wrapper enforces this:

```python
timeout = self._step_count >= self._max_steps
dones = [d or t or timeout for d, t in zip(done_base, trunc_base)]
```

When `all(dones)` is True, the episode ends. In practice almost all episodes hit the 500-step limit (there is no natural terminal state — robots can always keep going).

### Goal Randomization

With `randomize_goals: true`, packing station positions are randomized at the start of each episode:

```python
def _randomize_goal_positions(self):
    valid = [(c, r) for c in range(cols) for r in range(rows)
             if (c, r) not in shelf_positions]
    new_goals = random.sample(valid, len(u.goals))
    u.goals[:] = new_goals
```

Goals can appear anywhere in the grid except cells occupied by shelves. This prevents the policy from memorizing fixed goal locations and forces it to generalize — the agent must observe and navigate to the current goal position every episode.

### Shaping State Reset

All reward shaping state variables are reset at each `env.reset()`:

```python
self._prev_carrying   = [False] * n_agents   # was agent carrying last step?
self._prev_goal_dist  = [inf]   * n_agents   # distance to goal last step
self._prev_shelf_dist = [inf]   * n_agents   # distance to shelf last step
self._just_dropped    = [False] * n_agents   # did agent just drop a shelf?
self._drop_cooldown   = [0]     * n_agents   # steps remaining on drop cooldown
self._prev_positions  = [(0,0)] * n_agents   # for collision detection
```

---

## 10. Rendering

The wrapper provides a **headless matplotlib renderer** that returns an RGB numpy array:

```python
frame = env.render(cell_px=60)   # returns np.ndarray, shape (H, W, 3), dtype uint8
```

The renderer reads the rware internal state directly (`env._env.unwrapped`) and draws:

| Element | Visual | Color |
|---------|--------|-------|
| Empty cell | Background | `#F5F5F5` (off-white) |
| Shelf (uncarried) | Filled rectangle | `#90CAF9` (light blue) |
| Robot (empty-handed) | Circle + arrow | `#EF5350` (red) |
| Robot (carrying shelf) | Circle + arrow | `#66BB6A` (green) |
| Packing station | Rectangle + "P" label | `#FFF176` (yellow) |
| Grid lines | Lines | `#BDBDBD` (grey) |

The arrow inside each robot circle indicates the robot's facing direction (↑ → ↓ ←).

**Why custom rendering:** rware-v2's built-in `rgb_array` render mode is broken (returns incorrect frames). The matplotlib-based renderer was written as a reliable replacement.

**GIF recording:** Training scripts record a GIF of the best trained policy after training completes:

```python
frames.append(env.render())
imageio.mimsave("results/videos/mappo_trained.gif", frames, fps=8, loop=0)
```

---

## 11. Configuration Files

All environment parameters live in YAML files under `configs/`. The algorithm configs are separate (see `happo_implementation.md` or `research.md`).

### `configs/env_config.yaml` — 2-Agent

```yaml
env:
  name: "rware-tiny-2ag-v2"
  n_agents: 2
  max_steps: 500
  reward_type: "individual"
  randomize_goals: true
  reward_shaping:
    enabled: true
    pickup_reward: 0.5
    delivery_bonus: 2.0
    step_penalty: -0.005
    carry_toward_goal: 0.05
    move_toward_shelf: 0.05
    bad_drop_penalty: -0.6
    linger_penalty: -0.05
    collision_penalty: 0.0
  record:
    fps: 8
    output_dir: "results/videos"
    filename: "random_policy.gif"
```

### `configs/env_config_4ag.yaml` — 4-Agent

Same structure with:
- `name: "rware-small-4ag-v2"`, `n_agents: 4`
- Reduced shaping values (as described in Section 8)
- `output_dir: "results/videos_4ag"`

---

## 12. Agent Setup in Algorithms

### How Observations Are Consumed

Each algorithm reads `env.obs_dim` and `env.action_dim` to size its networks:

```python
env = WarehouseEnv(config)

# MAPPO: one shared actor/critic
mappo = MAPPO(mappo_config, obs_dim=env.obs_dim, action_dim=env.action_dim,
              n_agents=env.n_agents)

# HAPPO: per-agent actors, shared critic
happo = HAPPO(happo_config, obs_dim=env.obs_dim, action_dim=env.action_dim,
              n_agents=env.n_agents)
```

For both 2-agent and 4-agent rware variants: `obs_dim = 71`, `action_dim = 5`.

### Global State for the Critic

Both MAPPO and HAPPO construct a global state vector for the centralized critic by concatenating all agents' normalized observations plus a one-hot agent ID:

```
global_obs[i] = [obs_0 | obs_1 | ... | obs_{n-1} | one_hot(i)]
```

| Config | obs_dim | n_agents | global_obs_dim |
|--------|---------|----------|----------------|
| 2-agent | 71 | 2 | 71×2 + 2 = 144 |
| 4-agent | 71 | 4 | 71×4 + 4 = 288 |

This is the **Centralized Training** side of CTDE — the critic has access to full team information at training time. During execution, each agent only uses its own actor with its local `obs_dim=71` observation.

### Per-Step Data Flow

```
env.reset()
→ obs: List[np.ndarray]     shape: [n_agents × (obs_dim,)]

algorithm.select_actions(obs)
→ stack(obs)                shape: (n_agents, obs_dim)
→ normalize                 shape: (n_agents, obs_dim)     ← RunningMeanStd
→ actor(obs_i)              → action, log_prob  per agent
→ critic(global_obs_i)      → value             per agent
→ actions: (n_agents,)
→ log_probs: (n_agents,)
→ values: (n_agents,)

env.step(actions.tolist())
→ obs: List[np.ndarray]     next observations
→ rews: List[float]         shaped per-agent rewards
→ dones: List[bool]         per-agent done flags

buffer.insert(norm_obs, global_obs, actions, log_probs, rews, dones, values)
```

After `n_steps` of this loop, the buffer holds one complete rollout. GAE advantages are computed, then the PPO update runs.

### Parameter Sharing Comparison

```
MAPPO:
    actor  ──(shared weights)──►  agent 0
                              ──►  agent 1
                              ──►  ...

    critic ──(shared weights)──►  global_obs_0
                              ──►  global_obs_1
                              ──►  ...

HAPPO:
    actor_0 ──►  agent 0
    actor_1 ──►  agent 1
    ...

    critic  ──(shared weights)──►  global_obs_0  (same as MAPPO)
                              ──►  global_obs_1
                              ──►  ...
```

Parameter sharing in MAPPO means all agents always have exactly the same policy — useful for homogeneous agents, but limiting. HAPPO's independent actors can in principle develop specialized behaviors for different "roles" within the team, even if those roles emerge implicitly from training rather than being hardcoded.

---

## 13. Quick Start

```bash
# Install dependencies
pip install rware gymnasium imageio pyyaml torch

# Verify environment works
python scripts/check_env.py

# Record a random policy GIF
python scripts/record_gif.py

# Train MAPPO (2-agent)
python scripts/train_mappo.py

# Train HAPPO (2-agent)
python scripts/train_happo.py

# Train HAPPO (4-agent)
python scripts/train_happo.py \
  --env-config configs/env_config_4ag.yaml \
  --happo-config configs/happo_config_4ag.yaml

# Generate MAPPO vs random comparison report
python scripts/generate_report.py
```

### Environment Sanity Check

```python
from src.env.warehouse_env import WarehouseEnv
import yaml

config = yaml.safe_load(open("configs/env_config.yaml"))
env = WarehouseEnv(config)

print(f"n_agents   : {env.n_agents}")    # 2
print(f"obs_dim    : {env.obs_dim}")     # 71
print(f"action_dim : {env.action_dim}")  # 5

obs = env.reset()
print(f"obs shapes : {[o.shape for o in obs]}")  # [(71,), (71,)]

actions = [0, 2]   # agent 0 moves up, agent 1 moves down
obs, rews, dones, info = env.step(actions)
print(f"rewards    : {rews}")   # List[float], shaped rewards
print(f"dones      : {dones}")  # [False, False] (episode not over yet)
```
