"""
src/analytics/__init__.py — Reward aggregation and measurement for the random agent baseline.

Usage:
    tracker = RewardTracker(n_agents=2)
    tracker.start_episode()
    for each step:
        tracker.record_step(rewards)   # rewards: List[float]
    tracker.end_episode()
    tracker.save("results/logs/random_baseline.json")
    summary = tracker.summary()
"""

import json
import csv
import os
import time
from typing import List, Dict, Any


class RewardTracker:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.episodes: List[Dict[str, Any]] = []

        # Current episode state
        self._ep_steps: List[List[float]] = []   # per-step rewards (list of per-agent rewards)
        self._ep_start: float = 0.0

    def start_episode(self):
        self._ep_steps = []
        self._ep_start = time.time()

    def record_step(self, rewards: List[float]):
        """Record per-agent rewards for one step."""
        self._ep_steps.append(list(rewards))

    def end_episode(self):
        """Finalize current episode and append to history."""
        if not self._ep_steps:
            return

        ep_duration = time.time() - self._ep_start
        n_steps = len(self._ep_steps)

        # Total reward per agent over the episode
        agent_totals = [
            sum(self._ep_steps[t][i] for t in range(n_steps))
            for i in range(self.n_agents)
        ]
        team_total = sum(agent_totals)

        self.episodes.append({
            "episode": len(self.episodes),
            "n_steps": n_steps,
            "duration_s": round(ep_duration, 3),
            "agent_total_rewards": agent_totals,
            "team_total_reward": round(team_total, 4),
            "team_reward_per_step": round(team_total / n_steps, 6) if n_steps else 0.0,
            "step_rewards": self._ep_steps,        # full trace — useful for plotting
        })
        self._ep_steps = []

    def summary(self) -> Dict[str, Any]:
        """Aggregate stats across all recorded episodes."""
        if not self.episodes:
            return {}

        team_totals = [ep["team_total_reward"] for ep in self.episodes]
        n_steps_list = [ep["n_steps"] for ep in self.episodes]

        def _stats(values):
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            std = variance ** 0.5
            return {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        return {
            "n_episodes": len(self.episodes),
            "n_agents": self.n_agents,
            "team_total_reward": _stats(team_totals),
            "episode_length": _stats(n_steps_list),
            "per_agent_mean_total": [
                round(
                    sum(ep["agent_total_rewards"][i] for ep in self.episodes) / len(self.episodes),
                    4,
                )
                for i in range(self.n_agents)
            ],
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str, include_step_trace: bool = False):
        """
        Save results to JSON.
        Set include_step_trace=True to embed per-step rewards (large for long runs).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        episodes_out = []
        for ep in self.episodes:
            entry = {k: v for k, v in ep.items() if k != "step_rewards"}
            if include_step_trace:
                entry["step_rewards"] = ep["step_rewards"]
            episodes_out.append(entry)

        data = {
            "summary": self.summary(),
            "episodes": episodes_out,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[analytics] Saved reward log → {path}")

    def save_csv(self, path: str):
        """Save per-episode summary as CSV (convenient for spreadsheets / plots)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        fieldnames = [
            "episode", "n_steps", "duration_s",
            "team_total_reward", "team_reward_per_step",
        ] + [f"agent_{i}_total" for i in range(self.n_agents)]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep in self.episodes:
                row = {
                    "episode": ep["episode"],
                    "n_steps": ep["n_steps"],
                    "duration_s": ep["duration_s"],
                    "team_total_reward": ep["team_total_reward"],
                    "team_reward_per_step": ep["team_reward_per_step"],
                }
                for i, total in enumerate(ep["agent_total_rewards"]):
                    row[f"agent_{i}_total"] = round(total, 4)
                writer.writerow(row)

        print(f"[analytics] Saved CSV → {path}")
