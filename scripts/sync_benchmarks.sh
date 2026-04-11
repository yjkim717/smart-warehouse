#!/bin/bash
echo "Pulling canonical baseline JSON benchmark logs off remote branches..."
mkdir -p results/logs

git show main:results/logs/random_baseline_rewards.json > results/logs/random_baseline_rewards.json
git show main:results/logs/greedy_baseline_rewards.json > results/logs/greedy_baseline_rewards.json
git show mappo:results/logs/trained_policy_rewards.json > results/logs/trained_policy_rewards.json

# Copying over comparison reports directly just to have statistics tracked
git show mappo:results/reports/comparison_report.json > results/logs/mappo_comparison_report.json 2>/dev/null || true

echo "✓ Baseline cross-branch pulls successfully staged in results/logs!"
