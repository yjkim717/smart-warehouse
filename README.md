# Smart Warehouse — Multi-Agent Path Finding (MAPF)

> An Amazon Kiva-style warehouse simulation where a fleet of robots learn to coordinate shelf retrieval without collisions, maximizing throughput.

## Architecture

```
Centralized Training, Decentralized Execution (CTDE)
  - Training: Global state + joint observations → shared critic
  - Execution: Each robot acts on its own local observation only
```

## Project Structure

```
smart-warehouse/
├── src/
│   ├── env/          # Warehouse environment wrapper (rware)
│   ├── algorithms/   # MAPPO, QMIX implementations
│   ├── analytics/    # Heatmaps, bottleneck detection, metrics
│   └── utils/        # Logging, replay buffer, misc helpers
├── configs/          # Hyperparameter YAML files
├── scripts/          # Entry-point scripts (train, evaluate, visualize)
├── notebooks/        # EDA and result analysis
├── tests/            # Unit tests per module
└── results/          # Checkpoints, logs, plots, videos
```

## Phase 1 — Proof of Concept (Mar 12–16)

> "Robots move, learning beats random"

| File | Description |
|------|-------------|
| `configs/env_config.yaml` | Grid parameters — switch between tiny (2 robots) and small (4 robots) |
| `src/env/warehouse_env.py` | rware wrapper exposing `reset()`, `step()`, `render()`, `n_agents`, `obs_dim`, `action_dim` |
| `scripts/check_env.py` | Installation check + environment sanity test |
| `scripts/record_gif.py` | Record random or trained policy as GIF |

**Quick start:**

```bash
pip install rware gymnasium imageio pyyaml

python scripts/check_env.py            # verify install + env works
python scripts/record_gif.py           # record random policy GIF
python scripts/record_gif.py --checkpoint results/checkpoints/best_model.pt  # trained policy
```

**Interface for Team B:**

```python
from src.env.warehouse_env import WarehouseEnv
import yaml

with open("configs/env_config.yaml") as f:
    config = yaml.safe_load(f)

env = WarehouseEnv(config)
obs = env.reset()                           # List[np.ndarray], one per agent
obs, rews, dones, info = env.step(actions)  # actions: List[int]

env.n_agents    # number of robots
env.obs_dim     # observation vector size
env.action_dim  # 5 (up, right, down, left, interact)
```
