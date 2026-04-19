import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_rewards(filename, key='episodes', reward_key='team_total_reward'):
    with open(filename, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        if key in data:
            return [e[reward_key] for e in data[key]]
        elif '_rewards' in data:
            return data['_rewards']
    return []

def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Load data for 5 Reward Methods
methods_files = {
    'MAPPO': 'mappo_train_rewards.json',
    'HAPPO': 'happo_train_rewards.json',
    'QMIX+PER': 'qmix_per_train_rewards.json',
    'QMIX': 'qmix_train_rewards.json',
    'Greedy Baseline': 'greedy_baseline_rewards.json',
    'Random Baseline': 'random_baseline_rewards.json'
}

# Load Eval data
mappo_eval = pd.read_json('mappo_eval_curve.json')
happo_eval = pd.read_json('happo_eval_curve.json')
# QMIX data has 300 points which correspond to 10M episodes
with open('qmix_evaluation_rewards.json', 'r') as f:
    qmix_data = json.load(f)
qmix_rewards = qmix_data['_rewards']
# Generate x-axis for 10,000,000 episodes
qmix_x = np.linspace(0, 10000000, len(qmix_rewards))
qmix_per_eval = pd.read_json('qmix_per_eval_curve.json')

# 1. Reward Plot (5 Methods)
plt.figure(figsize=(12, 7))
window = 100

for name, file in methods_files.items():
    rew = load_rewards(file)
    if len(rew) > 0:
        smoothed = moving_average(rew, window)
        plt.plot(range(window-1, len(rew)), smoothed, label=name)

plt.xlabel('Episodes')
plt.ylabel('Team Total Reward (Moving Avg 100)')
plt.title('Training Reward Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('reward_comparison.png')

# 2. Evaluation Plot (3 Methods)
plt.figure(figsize=(10, 6))
plt.plot(mappo_eval['timestep'], mappo_eval['eval_mean_reward'], label='MAPPO')
plt.plot(happo_eval['timestep'], happo_eval['eval_mean_reward'], label='HAPPO')
# Plot standard QMIX on the 10M episode scale
plt.plot(qmix_x, qmix_rewards, label='QMIX', linewidth=2, color='grey')
plt.plot(qmix_per_eval['timestep'], qmix_per_eval['eval_mean_reward'], label='QMIX+PER')


# Greedy mean as a reference baseline
greedy_rew = load_rewards('greedy_baseline_rewards.json')
plt.axhline(y=np.mean(greedy_rew), color='r', linestyle='--', label='Greedy Baseline (Reference)')

plt.xlabel('Timesteps')
plt.ylabel('Evaluation Mean Reward')
plt.title('Evaluation Curve Comparison')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('eval_curve_comparison.png')