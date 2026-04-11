"""
Comparison plot: Random / Greedy / QMIX+PER (in-progress) / MAPPO
Uses terminal output data points captured from training run.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Baselines (from JSON logs) ──────────────────────────────────────────────
with open("results/logs/random_baseline_rewards.json") as f:
    random_mean = json.load(f)["summary"]["team_total_reward"]["mean"]

with open("results/logs/greedy_baseline_rewards.json") as f:
    greedy_mean = json.load(f)["summary"]["team_total_reward"]["mean"]

MAPPO_MEAN = 8.69

# ── QMIX+PER eval curve (captured from terminal output) ─────────────────────
# Format: (timestep, eval_mean_reward)
qmix_per_data = [
    (10000, 0.150), (20000, 0.175), (30000, 0.175), (40000, 0.125),
    (50000, 0.175), (60000, 0.250), (70000, 0.150), (80000, 0.075),
    (90000, 0.075), (100000, 0.100), (110000, 0.200), (120000, 0.125),
    (130000, 0.150), (140000, 0.200), (150000, 0.150), (160000, 0.275),
    (170000, 0.125), (180000, 0.075), (190000, 0.200), (200000, 0.225),
    (210000, 0.350), (220000, 0.150), (230000, 0.100), (240000, 0.275),
    (250000, 0.250), (260000, 0.150), (270000, 0.150), (280000, 0.225),
    (290000, 0.300), (300000, 0.225), (310000, 0.175), (320000, 0.150),
    (330000, 0.150), (340000, 0.175), (350000, 0.350), (360000, 0.225),
    (370000, 0.325), (380000, 0.100), (390000, 1.475), (400000, 0.175),
    (410000, 0.100), (420000, 0.100), (430000, 0.425), (440000, 0.250),
    (450000, 0.350), (460000, 0.300), (470000, 0.325), (480000, 0.075),
    (490000, 0.150), (500000, 0.125), (510000, 0.250), (520000, 0.200),
    (530000, 0.100), (540000, 0.325), (550000, 0.175), (560000, 0.200),
    (570000, 0.175), (580000, 0.300), (590000, 0.225), (600000, 0.275),
    (610000, 0.300), (620000, 0.200), (630000, 0.100), (640000, 0.150),
    (650000, 0.275), (660000, 0.150),
    # t=1.86M~2.01M
    (1860000, 0.425), (1870000, 0.350), (1880000, 1.875), (1890000, 0.450),
    (1900000, 0.600), (1910000, 0.575), (1920000, 0.525), (1930000, 0.750),
    (1940000, 0.425), (1950000, 0.375), (1960000, 0.450), (1970000, 0.500),
    (1980000, 0.775), (1990000, 0.650), (2000000, 0.400), (2010000, 0.525),
]

timesteps = np.array([d[0] for d in qmix_per_data])
rewards   = np.array([d[1] for d in qmix_per_data])

# Smoothed curve (window=5)
def smooth(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode='valid')

ts_smooth = timesteps[len(timesteps) - len(smooth(rewards)):]
r_smooth  = smooth(rewards)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

# Raw eval dots
ax.scatter(timesteps / 1e6, rewards, s=12, alpha=0.4, color="steelblue", label="QMIX+PER (eval, raw)")
# Smoothed line
ax.plot(ts_smooth / 1e6, r_smooth, color="steelblue", linewidth=2, label="QMIX+PER (smoothed)")

# Baselines
ax.axhline(random_mean, color="gray",   linestyle="--", linewidth=1.5, label=f"Random  ({random_mean:.3f})")
ax.axhline(greedy_mean, color="orange", linestyle="--", linewidth=1.5, label=f"Greedy  ({greedy_mean:.3f})")
ax.axhline(MAPPO_MEAN,  color="green",  linestyle="--", linewidth=1.5, label=f"MAPPO   ({MAPPO_MEAN:.2f})")

ax.set_xlabel("Timesteps (M)")
ax.set_ylabel("Eval Reward (team total)")
ax.set_title("Multi-Agent RL Comparison: rware-tiny-2ag-v2\n(QMIX+PER in progress, t=2M/3M)")
ax.legend(loc="upper left")
ax.set_ylim(-0.1, max(MAPPO_MEAN + 0.5, rewards.max() + 0.5))
ax.grid(True, alpha=0.3)

os.makedirs("results/plots", exist_ok=True)
out = "results/plots/comparison_qmix_per_vs_baselines.png"
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
