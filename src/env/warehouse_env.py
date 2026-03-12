"""
WarehouseEnv — rware wrapper for MAPPO/QMIX training.

Interface (what Team B needs):
    env = WarehouseEnv(config)
    obs = env.reset()                        # List[np.ndarray], length = n_agents
    obs, rews, dones, info = env.step(actions)  # actions: List[int]
    env.n_agents, env.obs_dim, env.action_dim
    env.render()                             # returns RGB array (H x W x 3, uint8)
"""

import random
import rware  # registers rware envs with gymnasium
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display window needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any

# Agent direction arrows
_DIR_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

# Colors
_COLOR_EMPTY   = "#F5F5F5"
_COLOR_SHELF   = "#90CAF9"   # light blue — shelf with package
_COLOR_AGENT   = "#EF5350"   # red — robot
_COLOR_CARRIED = "#66BB6A"   # green — robot carrying shelf
_COLOR_GOAL    = "#FFF176"   # yellow — delivery station
_COLOR_GRID    = "#BDBDBD"


class WarehouseEnv:
    """
    Thin wrapper around rware that:
    - Exposes a consistent multi-agent interface for MAPPO/QMIX
    - Renders via matplotlib (rware-v2 rgb_array is broken)
    """

    def __init__(self, config: dict):
        env_cfg = config["env"]
        env_name = env_cfg["name"]  # e.g. "rware-tiny-2ag-v2"

        # No render_mode — we render ourselves via matplotlib
        self._env = gym.make(env_name)
        self._n_agents: int = env_cfg["n_agents"]
        self._max_steps: int = env_cfg.get("max_steps", 500)
        self._step_count: int = 0
        self._randomize_goals: bool = env_cfg.get("randomize_goals", False)

        obs_space = self._env.observation_space
        act_space = self._env.action_space

        if hasattr(obs_space, "spaces"):
            self._obs_dim: int = obs_space.spaces[0].shape[0]
            self._action_dim: int = act_space.spaces[0].n
        else:
            self._obs_dim = obs_space.shape[0]
            self._action_dim = act_space.n

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> List[np.ndarray]:
        """Reset env, return list of per-agent observations."""
        self._step_count = 0
        obs, _ = self._env.reset()
        if self._randomize_goals:
            self._randomize_goal_positions()
        return self._unpack_obs(obs)

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """
        Step the environment.

        Args:
            actions: one integer action per agent

        Returns:
            obs   : List[np.ndarray]  — per-agent observations
            rews  : List[float]       — per-agent rewards
            dones : List[bool]        — per-agent done flags
            info  : dict
        """
        self._step_count += 1
        obs, rews, terminated, truncated, info = self._env.step(actions)

        obs_list = self._unpack_obs(obs)
        rew_list = self._unpack_scalar(rews)

        done_base = self._unpack_bool(terminated)
        trunc_base = self._unpack_bool(truncated)
        timeout = self._step_count >= self._max_steps
        dones = [d or t or timeout for d, t in zip(done_base, trunc_base)]

        return obs_list, rew_list, dones, info

    def render(self, cell_px: int = 60) -> np.ndarray:
        """
        Render current state as an RGB numpy array using matplotlib.
        Reads rware internal state directly (works with rware-v2).

        Returns:
            np.ndarray of shape (H, W, 3), dtype uint8
        """
        u = self._env.unwrapped

        grid = u.grid  # shape: (layers, rows, cols)
        rows, cols = grid.shape[1], grid.shape[2]

        dpi = 100
        fig_w = cols * cell_px / dpi
        fig_h = rows * cell_px / dpi

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect("equal")
        ax.axis("off")

        ax.set_facecolor(_COLOR_EMPTY)
        fig.patch.set_facecolor(_COLOR_EMPTY)
        font_size = max(8, cell_px * 0.2)

        # Grid lines
        for x in range(cols + 1):
            ax.axvline(x, color=_COLOR_GRID, linewidth=0.5)
        for y in range(rows + 1):
            ax.axhline(y, color=_COLOR_GRID, linewidth=0.5)

        # Draw packing stations (goals)
        for (gc, gr) in u.goals:
            rect = patches.Rectangle(
                (gc, rows - gr - 1), 1, 1,
                linewidth=0, facecolor=_COLOR_GOAL, alpha=0.9
            )
            ax.add_patch(rect)
            ax.text(
                gc + 0.5, rows - gr - 0.5, "P",
                ha="center", va="center",
                fontsize=font_size, color="#795548", fontweight="bold", zorder=2
            )

        # Draw shelves
        for shelf in u.shelfs:
            r, c = shelf.y, shelf.x
            rect = patches.Rectangle(
                (c, rows - r - 1), 1, 1,
                linewidth=0, facecolor=_COLOR_SHELF, alpha=0.7
            )
            ax.add_patch(rect)

        # Draw agents
        for agent in u.agents:
            r, c = agent.y, agent.x
            color = _COLOR_CARRIED if agent.carrying_shelf else _COLOR_AGENT
            circle = plt.Circle(
                (c + 0.5, rows - r - 0.5), 0.35,
                color=color, zorder=3
            )
            ax.add_patch(circle)
            arrow = _DIR_ARROWS.get(agent.dir.value, "?")
            ax.text(
                c + 0.5, rows - r - 0.5, arrow,
                ha="center", va="center",
                fontsize=font_size,
                color="white", fontweight="bold", zorder=4
            )

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame = buf.reshape(h, w, 4)[:, :, :3]  # RGBA → RGB
        plt.close(fig)
        return frame

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Properties (used by Team B to build networks)
    # ------------------------------------------------------------------

    @property
    def n_agents(self) -> int:
        return self._n_agents

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _randomize_goal_positions(self):
        """Randomly relocate packing stations to positions not occupied by shelves."""
        u = self._env.unwrapped
        rows, cols = u.grid.shape[1], u.grid.shape[2]

        shelf_positions = {(s.x, s.y) for s in u.shelfs}
        valid = [(c, r) for c in range(cols) for r in range(rows)
                 if (c, r) not in shelf_positions]

        new_goals = random.sample(valid, len(u.goals))
        u.goals[:] = new_goals

    def _unpack_obs(self, obs) -> List[np.ndarray]:
        if isinstance(obs, (list, tuple)):
            return [np.array(o, dtype=np.float32) for o in obs]
        return [np.array(obs[i], dtype=np.float32) for i in range(self._n_agents)]

    def _unpack_scalar(self, vals) -> List[float]:
        if isinstance(vals, (list, tuple, np.ndarray)):
            return [float(v) for v in vals]
        return [float(vals)] * self._n_agents

    def _unpack_bool(self, vals) -> List[bool]:
        if isinstance(vals, (list, tuple, np.ndarray)):
            return [bool(v) for v in vals]
        return [bool(vals)] * self._n_agents
