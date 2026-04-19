"""
generate_happo_report.py — HAPPO vs Random Policy comparison report.

Runs both policies for N episodes under identical conditions, then produces:
  - results/reports_happo/comparison_report.txt   human-readable summary
  - results/reports_happo/comparison_report.json  machine-readable data
  - results/reports_happo/comparison_plots.png    6-panel figure

Usage:
    python scripts/generate_happo_report.py
    python scripts/generate_happo_report.py --episodes 300 --steps 500
    python scripts/generate_happo_report.py --env-config configs/env_config_4ag.yaml \\
        --happo-config configs/happo_config_4ag.yaml
    python scripts/generate_happo_report.py --checkpoint results/checkpoints_happo/best_model.pt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, ".")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stats(arr):
    a = np.array(arr, dtype=float)
    return {
        "mean":   round(float(a.mean()), 4),
        "std":    round(float(a.std()),  4),
        "median": round(float(np.median(a)), 4),
        "min":    round(float(a.min()),  4),
        "max":    round(float(a.max()),  4),
        "p25":    round(float(np.percentile(a, 25)), 4),
        "p75":    round(float(np.percentile(a, 75)), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# HAPPO evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_happo(env_cfg: dict, checkpoint_path: str, n_episodes: int, max_steps: int) -> dict:
    import torch
    from src.env.warehouse_env import WarehouseEnv
    from src.algorithms.networks import Actor

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]

    # Verify this is a HAPPO checkpoint
    algo = meta.get("algo", "unknown")
    if algo != "happo":
        print(f"  [warn] checkpoint metadata says algo='{algo}', expected 'happo'")

    n_agents   = meta["n_agents"]
    obs_dim    = meta["obs_dim"]
    action_dim = meta["action_dim"]
    hidden_dim = meta["hidden_dim"]
    n_layers   = meta["n_layers"]

    # Load one actor per agent
    actors = []
    for i, state_dict in enumerate(ckpt["actors"]):
        actor = Actor(obs_dim, action_dim, hidden_dim, n_layers)
        actor.load_state_dict(state_dict)
        actor.eval()
        actors.append(actor)

    obs_mean = ckpt.get("obs_rms", {}).get("mean", np.zeros(obs_dim))
    obs_var  = ckpt.get("obs_rms", {}).get("var",  np.ones(obs_dim))

    env = WarehouseEnv(env_cfg)
    assert env.n_agents == n_agents, (
        f"Checkpoint has {n_agents} agents but env has {env.n_agents}. "
        f"Use the matching env config."
    )

    episode_rewards    = []
    episode_lengths    = []
    episode_deliveries = []
    episode_pickups    = []
    per_agent_rewards  = [[] for _ in range(n_agents)]

    print(f"  [happo] evaluating {n_episodes} episodes with {n_agents} independent actors …")
    t0 = time.time()

    for ep in range(n_episodes):
        obs = env.reset()
        u = env._env.unwrapped
        total_shaped  = 0.0
        agent_totals  = np.zeros(n_agents)
        deliveries    = 0
        pickups       = 0
        prev_carrying = [False] * n_agents

        for step in range(max_steps):
            raw  = np.stack(obs).astype(np.float64)
            norm = ((raw - obs_mean) / (np.sqrt(obs_var) + 1e-8)).astype(np.float32)

            actions = []
            with torch.no_grad():
                for i in range(n_agents):
                    obs_i  = torch.tensor(norm[i:i+1])
                    logits = actors[i](obs_i)
                    # Greedy argmax (deterministic eval)
                    actions.append(logits.argmax(dim=-1).item())

            obs, rews, dones, _ = env.step(actions)

            # Count pickups from carrying state change
            for i, agent in enumerate(u.agents):
                now_carrying = bool(agent.carrying_shelf)
                if now_carrying and not prev_carrying[i]:
                    pickups += 1
                prev_carrying[i] = now_carrying

            # Delivery detection: shaped reward > 0.9 catches base(1.0) + bonus - step penalty
            for r in rews:
                if r > 0.9:
                    deliveries += 1

            total_shaped  += sum(rews)
            agent_totals  += np.array(rews)

            if all(dones):
                break

        episode_rewards.append(total_shaped)
        episode_lengths.append(step + 1)
        episode_deliveries.append(deliveries)
        episode_pickups.append(pickups)
        for i in range(n_agents):
            per_agent_rewards[i].append(agent_totals[i])

        if (ep + 1) % 100 == 0:
            print(f"    ep {ep+1}/{n_episodes}  mean_so_far={np.mean(episode_rewards):.3f}")

    env.close()
    elapsed = time.time() - t0

    positive = sum(1 for r in episode_rewards if r > 0)
    return {
        "policy":            f"HAPPO ({n_agents} agents)",
        "n_agents":          n_agents,
        "n_episodes":        n_episodes,
        "eval_time_s":       round(elapsed, 1),
        "reward":            stats(episode_rewards),
        "episode_length":    stats(episode_lengths),
        "deliveries_per_ep": stats(episode_deliveries),
        "pickups_per_ep":    stats(episode_pickups),
        "positive_rate":     round(positive / n_episodes, 4),
        "positive_count":    positive,
        "per_agent_reward":  [stats(per_agent_rewards[i]) for i in range(n_agents)],
        # raw arrays for plots (stripped before JSON save)
        "_rewards":          episode_rewards,
        "_deliveries":       episode_deliveries,
        "_lengths":          episode_lengths,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Random policy evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_random(env_cfg: dict, n_episodes: int, max_steps: int) -> dict:
    from src.env.warehouse_env import WarehouseEnv

    # Disable shaping for random — keeps rewards on the raw rware scale
    cfg = yaml.safe_load(yaml.dump(env_cfg))   # deep copy
    cfg["env"]["reward_shaping"]["enabled"] = False

    env = WarehouseEnv(cfg)
    n_agents = env.n_agents

    episode_rewards    = []
    episode_lengths    = []
    episode_deliveries = []
    episode_pickups    = []
    per_agent_rewards  = [[] for _ in range(n_agents)]

    print(f"  [random] evaluating {n_episodes} episodes …")
    t0 = time.time()

    for ep in range(n_episodes):
        obs = env.reset()
        u = env._env.unwrapped
        total        = 0.0
        agent_totals = np.zeros(n_agents)
        deliveries   = 0
        pickups      = 0
        prev_carrying = [False] * n_agents

        for step in range(max_steps):
            goal_set  = {(gc, gr) for gc, gr in u.goals}
            shelf_set = {(s.x, s.y) for s in u.shelfs}
            actions   = []
            for agent in u.agents:
                pos = (agent.x, agent.y)
                if agent.carrying_shelf and pos in goal_set:
                    actions.append(4)
                elif agent.carrying_shelf:
                    actions.append(np.random.randint(4))
                elif not agent.carrying_shelf and pos in goal_set:
                    actions.append(np.random.randint(4))
                elif not agent.carrying_shelf and pos in shelf_set:
                    actions.append(4)
                else:
                    actions.append(np.random.randint(5))

            obs, rews, dones, _ = env.step(actions)

            for i, agent in enumerate(u.agents):
                now_carrying = bool(agent.carrying_shelf)
                if now_carrying and not prev_carrying[i]:
                    pickups += 1
                prev_carrying[i] = now_carrying

            for r in rews:
                if r > 0:
                    deliveries += 1

            total        += sum(rews)
            agent_totals += np.array(rews)

            if all(dones):
                break

        episode_rewards.append(total)
        episode_lengths.append(step + 1)
        episode_deliveries.append(deliveries)
        episode_pickups.append(pickups)
        for i in range(n_agents):
            per_agent_rewards[i].append(agent_totals[i])

        if (ep + 1) % 100 == 0:
            print(f"    ep {ep+1}/{n_episodes}  mean_so_far={np.mean(episode_rewards):.3f}")

    env.close()
    elapsed = time.time() - t0

    positive = sum(1 for r in episode_rewards if r > 0)
    return {
        "policy":            "Random (semi-greedy)",
        "n_agents":          n_agents,
        "n_episodes":        n_episodes,
        "eval_time_s":       round(elapsed, 1),
        "reward":            stats(episode_rewards),
        "episode_length":    stats(episode_lengths),
        "deliveries_per_ep": stats(episode_deliveries),
        "pickups_per_ep":    stats(episode_pickups),
        "positive_rate":     round(positive / n_episodes, 4),
        "positive_count":    positive,
        "per_agent_reward":  [stats(per_agent_rewards[i]) for i in range(n_agents)],
        "_rewards":          episode_rewards,
        "_deliveries":       episode_deliveries,
        "_lengths":          episode_lengths,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ──────────────────────────────────────────────────────────────────────────────

def welch_t_test(a, b):
    from scipy import stats as sp_stats
    t, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return float((a.mean() - b.mean()) / (pooled_std + 1e-9))


# ──────────────────────────────────────────────────────────────────────────────
# Text report
# ──────────────────────────────────────────────────────────────────────────────

def write_text_report(happo: dict, rand: dict, stats_test: dict, out_path: str,
                      eval_curve_path: str) -> str:
    lines = []
    W = 64
    n_agents = happo["n_agents"]

    def hr(char="─"):
        lines.append(char * W)

    def section(title):
        hr("═")
        lines.append(f"  {title}")
        hr("═")

    def row(label, h_val, r_val, higher_better=True):
        if isinstance(h_val, float):
            better = h_val > r_val if higher_better else h_val < r_val
            flag = "◀ better" if better else ""
            lines.append(f"  {label:<28}  {h_val:>10.4f}   {r_val:>10.4f}  {flag}")
        else:
            lines.append(f"  {label:<28}  {str(h_val):>10}   {str(r_val):>10}")

    header = [
        "",
        "=" * W,
        f"  SMART WAREHOUSE — HAPPO vs Random Policy Comparison Report",
        "=" * W,
        f"  Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Agents    : {n_agents}",
        f"  Episodes  : {happo['n_episodes']} per policy",
        f"  Max steps : 500 per episode",
        f"  HAPPO     : greedy argmax, independent per-agent actors, WITH reward shaping",
        f"  Random    : semi-greedy (auto-pickup/drop), raw rware reward (no shaping)",
        "",
        "  NOTE: reward scales differ — HAPPO shaping adds ~+0.5 per delivery,",
        "  +0.1 per pickup, −step_penalty. Use deliveries_per_ep for fair comparison.",
        "",
    ]
    lines.extend(header)

    # ── 1. Team reward ──────────────────────────────────────────────────────
    section("1. Team Total Reward (per episode)")
    lines.append(f"  {'Metric':<28}  {'HAPPO':>10}   {'Random':>10}")
    hr()
    for key, label, hb in [
        ("mean",   "Mean",        True),
        ("std",    "Std Dev",     False),
        ("median", "Median",      True),
        ("min",    "Min",         True),
        ("max",    "Max",         True),
        ("p25",    "25th pctile", True),
        ("p75",    "75th pctile", True),
    ]:
        row(label, happo["reward"][key], rand["reward"][key], hb)
    lines.append("")
    pos_h = f"{happo['positive_count']}/{happo['n_episodes']} ({happo['positive_rate']*100:.1f}%)"
    pos_r = f"{rand['positive_count']}/{rand['n_episodes']} ({rand['positive_rate']*100:.1f}%)"
    lines.append(f"  {'Positive episodes':<28}  {pos_h:>10}   {pos_r:>10}")

    # ── 2. Deliveries ───────────────────────────────────────────────────────
    section("2. Deliveries per Episode  [same raw scale for both]")
    lines.append(f"  {'Metric':<28}  {'HAPPO':>10}   {'Random':>10}")
    hr()
    for key, label in [
        ("mean",   "Mean deliveries"),
        ("std",    "Std Dev"),
        ("median", "Median"),
        ("max",    "Max in one episode"),
    ]:
        row(label, happo["deliveries_per_ep"][key], rand["deliveries_per_ep"][key])
    lines.append("")
    h_any = sum(1 for d in happo["_deliveries"] if d > 0)
    r_any = sum(1 for d in rand["_deliveries"]  if d > 0)
    lines.append(f"  {'Episodes with ≥1 delivery':<28}  {h_any:>10}   {r_any:>10}")
    lines.append(f"  {'Delivery rate':<28}  "
                 f"{h_any/happo['n_episodes']*100:>9.1f}%   "
                 f"{r_any/rand['n_episodes']*100:>9.1f}%")

    # ── 3. Pickups ──────────────────────────────────────────────────────────
    section("3. Pickups per Episode")
    lines.append(f"  {'Metric':<28}  {'HAPPO':>10}   {'Random':>10}")
    hr()
    for key, label in [("mean", "Mean pickups"), ("std", "Std Dev"), ("max", "Max")]:
        row(label, happo["pickups_per_ep"][key], rand["pickups_per_ep"][key])

    # ── 4. Episode length ───────────────────────────────────────────────────
    section("4. Episode Length (steps)")
    lines.append(f"  {'Metric':<28}  {'HAPPO':>10}   {'Random':>10}")
    hr()
    for key, label in [("mean", "Mean"), ("std", "Std Dev"), ("min", "Min"), ("max", "Max")]:
        row(label, happo["episode_length"][key], rand["episode_length"][key], False)

    # ── 5. Per-agent rewards ─────────────────────────────────────────────────
    section("5. Per-Agent Mean Reward  [HAPPO: independent actors]")
    lines.append(f"  {'Agent':<28}  {'HAPPO':>10}   {'Random':>10}")
    hr()
    for i, (h_ar, r_ar) in enumerate(zip(happo["per_agent_reward"], rand["per_agent_reward"])):
        row(f"Agent {i}", h_ar["mean"], r_ar["mean"])

    # ── 6. Statistical significance ─────────────────────────────────────────
    section("6. Statistical Significance")
    t, p = stats_test["t"], stats_test["p"]
    d    = stats_test["cohens_d"]
    sig  = "YES (p < 0.001)" if p < 0.001 else ("YES (p < 0.05)" if p < 0.05 else "NO")
    mag  = "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small")
    lines += [
        f"  Welch's t-test (reward):",
        f"    t-statistic : {t:.4f}",
        f"    p-value     : {p:.2e}",
        f"    Significant : {sig}",
        f"  Cohen's d    : {d:.4f}  ({mag} effect)",
        "",
        f"  Interpretation: HAPPO's reward distribution is statistically",
        f"  {'different from' if p < 0.05 else 'NOT significantly different from'} "
        f"random (p={p:.2e}). Effect size is {mag}.",
    ]

    # ── 7. Training curve summary ────────────────────────────────────────────
    section("7. HAPPO Training Curve Summary")
    try:
        with open(eval_curve_path) as f:
            curve = json.load(f)
        # Support both list-of-dicts and dict-of-lists formats
        if isinstance(curve, list):
            rew_curve = [e["eval_mean_reward"] for e in curve]
            ts_curve  = [e["timestep"] for e in curve]
            ent_curve = [e.get("entropy", float("nan")) for e in curve]
        else:
            rew_curve = curve.get("eval_mean_rewards", curve.get("mean_rewards", []))
            ts_curve  = curve.get("timesteps", list(range(len(rew_curve))))
            ent_curve = curve.get("entropies", [float("nan")] * len(rew_curve))
        n       = len(rew_curve)
        first5  = np.mean(rew_curve[:5])  if n >= 5 else np.mean(rew_curve)
        last5   = np.mean(rew_curve[-5:]) if n >= 5 else np.mean(rew_curve)
        best    = max(rew_curve)
        best_t  = ts_curve[rew_curve.index(best)]
        pos_n   = sum(1 for r in rew_curve if r > 0)
        lines += [
            f"  Eval checkpoints   : {n}",
            f"  Timesteps trained  : {ts_curve[-1]:,}",
            f"  Reward (first 5 ck): {first5:.4f}",
            f"  Reward (last 5 ck) : {last5:.4f}",
            f"  Best eval reward   : {best:.4f}  at t={best_t:,}",
            f"  Positive evals     : {pos_n}/{n}",
            f"  Start entropy      : {ent_curve[0]:.4f}",
            f"  End entropy        : {ent_curve[-1]:.4f}  (lower = more decisive)",
        ]
    except Exception as e:
        lines.append(f"  [could not load eval curve: {e}]")

    # ── 8. Verdict ──────────────────────────────────────────────────────────
    section("8. Verdict")
    reward_lift   = happo["reward"]["mean"] - rand["reward"]["mean"]
    delivery_lift = happo["deliveries_per_ep"]["mean"] - rand["deliveries_per_ep"]["mean"]
    h_delivery_rate = sum(1 for d in happo["_deliveries"] if d > 0) / happo["n_episodes"] * 100
    r_delivery_rate = sum(1 for d in rand["_deliveries"]  if d > 0) / rand["n_episodes"]  * 100
    lines += [
        f"  HAPPO ({n_agents} agents) vs Random over {happo['n_episodes']} episodes each:",
        "",
        f"  Mean reward lift    : {reward_lift:+.3f} (shaped vs raw — not directly comparable)",
        f"  Positive rate       : {happo['positive_rate']*100:.1f}% vs {rand['positive_rate']*100:.1f}%",
        f"  Delivery rate       : {h_delivery_rate:.1f}% vs {r_delivery_rate:.1f}% (eps with ≥1 delivery)",
        f"  Avg deliveries/ep   : {happo['deliveries_per_ep']['mean']:.3f} vs {rand['deliveries_per_ep']['mean']:.3f}  ({delivery_lift:+.3f})",
        f"  Statistical test    : {sig}",
        "",
        f"  CONCLUSION: HAPPO {'significantly' if p < 0.05 else 'does not significantly'} "
        f"outperforms the random baseline.",
        "  Use 'deliveries_per_ep' as the primary fair metric (same scale for both).",
    ]

    hr("═")
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def generate_plots(happo: dict, rand: dict, out_path: str, eval_curve_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAPPO_COLOR = "#1976D2"
    RAND_COLOR  = "#EF5350"
    ALPHA       = 0.75

    h_r = np.array(happo["_rewards"])
    r_r = np.array(rand["_rewards"])
    h_d = np.array(happo["_deliveries"])
    r_d = np.array(rand["_deliveries"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"HAPPO ({happo['n_agents']} agents) vs Random Policy — Comparison Report",
        fontsize=15, fontweight="bold"
    )

    # ── Panel 1: Reward distribution ────────────────────────────────────────
    ax = axes[0, 0]
    all_vals = np.concatenate([h_r, r_r])
    bins = np.linspace(all_vals.min() - 0.5, all_vals.max() + 0.5, 40)
    ax.hist(h_r, bins=bins, alpha=ALPHA, color=HAPPO_COLOR, label=f"HAPPO (μ={h_r.mean():.2f})")
    ax.hist(r_r, bins=bins, alpha=ALPHA, color=RAND_COLOR,  label=f"Random (μ={r_r.mean():.2f})")
    ax.axvline(h_r.mean(), color=HAPPO_COLOR, linewidth=2, linestyle="--")
    ax.axvline(r_r.mean(), color=RAND_COLOR,  linewidth=2, linestyle="--")
    ax.set_xlabel("Team Total Reward per Episode")
    ax.set_ylabel("Episode Count")
    ax.set_title("Reward Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel 2: Box plot ────────────────────────────────────────────────────
    ax = axes[0, 1]
    bp = ax.boxplot(
        [h_r, r_r],
        labels=["HAPPO", "Random"],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    bp["boxes"][0].set_facecolor(HAPPO_COLOR)
    bp["boxes"][1].set_facecolor(RAND_COLOR)
    ax.set_ylabel("Team Total Reward per Episode")
    ax.set_title("Reward Box Plot")
    ax.grid(alpha=0.3, axis="y")

    # ── Panel 3: Deliveries bar ──────────────────────────────────────────────
    ax = axes[0, 2]
    labels = ["HAPPO", "Random"]
    means  = [h_d.mean(), r_d.mean()]
    stds   = [h_d.std(),  r_d.std()]
    colors = [HAPPO_COLOR, RAND_COLOR]
    bars = ax.bar(labels, means, color=colors, alpha=ALPHA, yerr=stds, capsize=8)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + max(stds) * 0.05,
                f"{m:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Deliveries per Episode")
    ax.set_title("Mean Deliveries per Episode ± Std")
    ax.grid(alpha=0.3, axis="y")

    # ── Panel 4: Training curve ──────────────────────────────────────────────
    ax = axes[1, 0]
    try:
        with open(eval_curve_path) as f:
            curve = json.load(f)
        if isinstance(curve, list):
            ts   = [e["timestep"] for e in curve]
            rew  = [e["eval_mean_reward"] for e in curve]
            ent  = [e.get("entropy", float("nan")) for e in curve]
        else:
            ts   = curve.get("timesteps", [])
            rew  = curve.get("eval_mean_rewards", curve.get("mean_rewards", []))
            ent  = curve.get("entropies", [float("nan")] * len(rew))

        ax.plot(ts, rew, color=HAPPO_COLOR, linewidth=1.5, label="HAPPO eval reward")
        ax.axhline(rand["reward"]["mean"], color=RAND_COLOR, linestyle="--",
                   linewidth=1.2, label=f"Random mean ({rand['reward']['mean']:.3f})")
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Training Timestep")
        ax.set_ylabel("Mean Eval Reward")
        ax.set_title("HAPPO Training Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(ts, ent, color="#AB47BC", linewidth=1, linestyle="--", alpha=0.6)
        ax2.set_ylabel("Entropy", color="#AB47BC")
        ax2.tick_params(axis="y", labelcolor="#AB47BC")
    except Exception as e:
        ax.text(0.5, 0.5, f"No eval curve:\n{e}", ha="center", va="center",
                transform=ax.transAxes)

    # ── Panel 5: Cumulative deliveries ───────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(np.cumsum(h_d), color=HAPPO_COLOR, linewidth=1.5, label="HAPPO")
    ax.plot(np.cumsum(r_d), color=RAND_COLOR,  linewidth=1.5, label="Random")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Deliveries")
    ax.set_title("Cumulative Deliveries over Episodes")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel 6: Per-agent reward bars ───────────────────────────────────────
    ax = axes[1, 2]
    n_agents = happo["n_agents"]
    x = np.arange(n_agents)
    w = 0.35
    h_means = [happo["per_agent_reward"][i]["mean"] for i in range(n_agents)]
    r_means = [rand["per_agent_reward"][i]["mean"]  for i in range(n_agents)]
    h_stds  = [happo["per_agent_reward"][i]["std"]  for i in range(n_agents)]
    r_stds  = [rand["per_agent_reward"][i]["std"]   for i in range(n_agents)]
    ax.bar(x - w/2, h_means, w, color=HAPPO_COLOR, alpha=ALPHA, label="HAPPO",
           yerr=h_stds, capsize=5)
    ax.bar(x + w/2, r_means, w, color=RAND_COLOR,  alpha=ALPHA, label="Random",
           yerr=r_stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Agent {i}" for i in range(n_agents)])
    ax.set_ylabel("Mean Reward per Episode")
    ax.set_title("Per-Agent Mean Reward ± Std")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[report] Plot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate HAPPO vs Random comparison report"
    )
    parser.add_argument("--env-config",   default="configs/env_config.yaml",
                        help="Path to env config yaml")
    parser.add_argument("--happo-config", default="configs/happo_config.yaml",
                        help="Path to happo config yaml — used to resolve log_dir and checkpoint_dir")
    parser.add_argument("--checkpoint",   default=None,
                        help="Path to HAPPO checkpoint (default: checkpoint_dir/best_model.pt)")
    parser.add_argument("--episodes",     type=int, default=300,
                        help="Episodes per policy (default 300)")
    parser.add_argument("--steps",        type=int, default=500,
                        help="Max steps per episode (default 500)")
    args = parser.parse_args()

    env_cfg   = load_config(args.env_config)
    happo_cfg = load_config(args.happo_config)
    log_dir   = happo_cfg["happo"]["log_dir"]
    ckpt_dir  = happo_cfg["happo"]["checkpoint_dir"]

    # Derive report dir from log dir: results/logs_happo → results/reports_happo
    report_dir = log_dir.replace("logs", "reports")

    checkpoint_path = args.checkpoint or os.path.join(ckpt_dir, "best_model.pt")
    eval_curve_path = os.path.join(log_dir, "happo_eval_curve.json")
    txt_out         = os.path.join(report_dir, "comparison_report.txt")
    json_out        = os.path.join(report_dir, "comparison_report.json")
    plot_out        = os.path.join(report_dir, "comparison_plots.png")

    print("=" * 58)
    print("  Smart Warehouse — HAPPO vs Random Comparison Report")
    print("=" * 58)
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Env config : {args.env_config}")
    print(f"  Episodes   : {args.episodes} per policy")
    print()

    happo_results = eval_happo(env_cfg, checkpoint_path, args.episodes, args.steps)
    rand_results  = eval_random(env_cfg, args.episodes, args.steps)

    # Statistical tests
    try:
        t, p = welch_t_test(happo_results["_rewards"], rand_results["_rewards"])
    except ImportError:
        print("  [warn] scipy not installed — skipping t-test (pip install scipy)")
        t, p = float("nan"), float("nan")
    d = cohens_d(happo_results["_rewards"], rand_results["_rewards"])
    stats_test = {"t": round(t, 4), "p": p, "cohens_d": round(d, 4)}

    # Save JSON (strip raw arrays)
    os.makedirs(report_dir, exist_ok=True)
    happo_clean = {k: v for k, v in happo_results.items() if not k.startswith("_")}
    rand_clean  = {k: v for k, v in rand_results.items()  if not k.startswith("_")}
    with open(json_out, "w") as f:
        json.dump({"happo": happo_clean, "random": rand_clean, "statistics": stats_test},
                  f, indent=2)
    print(f"[report] JSON saved → {json_out}")

    generate_plots(happo_results, rand_results, plot_out, eval_curve_path)

    text = write_text_report(happo_results, rand_results, stats_test, txt_out, eval_curve_path)
    print()
    print(text)
    print(f"\n[report] Text report saved → {txt_out}")


if __name__ == "__main__":
    main()
