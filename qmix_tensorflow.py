import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import copy

# Add ENV SIMULATION
import yaml
from src.env.warehouse_env import WarehouseEnv

# 1. Load the warehouse configuration (e.g., grid size, agents)
with open("configs/env_config.yaml") as f:
    config = yaml.safe_load(f)

# 2. Instantiate the environment
env = WarehouseEnv(config)

# 3. Pull the dynamic dimensions straight from the environment!
N_AGENTS = int(env.n_agents)
N_ACTIONS = int(env.action_dim)
N_OBS_DIM = int(env.obs_dim)

# Recompute N_FEATURES 
N_FEATURES = N_AGENTS + N_OBS_DIM + N_ACTIONS

EYE_AGENTS = np.eye(N_AGENTS, dtype=np.float32)
EYE_ACTIONS = np.eye(N_ACTIONS, dtype=np.float32)

# Sync with MAPPO Hyperparameters
HIDDEN_DIM = 128
MIXER_HIDDEN_DIM = 128
NUM_EPISODES = 3000000
MAX_STEPS = config['env'].get('max_steps', 500)
GAMMA = 0.99
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_INTERVAL = 20  # Update every 20 episodes (~1000 gradient steps), prevents catastrophic forgetting

# Epsilon Scheduling instead of fixed
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = int(NUM_EPISODES * 0.7)

class QMixAgent(tf.keras.Model):
    def __init__(self, hidden_dim, n_actions):
        super(QMixAgent, self).__init__()
        self.hidden_dim = hidden_dim

        # Dense Feed-Forward network eliminating GRU sequential shuffle bugs
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(n_actions)

    @tf.function(reduce_retracing=True)
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        q_values = self.fc3(x)
        return q_values


class QMixer(tf.keras.Model):
    def __init__(self, n_agents, state_dim, embed_dim, hypernet_embed_dim):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # --- Layer 1 hypernetworks ---
        self.hyper_w_1 = tf.keras.Sequential([
            tf.keras.layers.Dense(hypernet_embed_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim * n_agents)
        ])
        
        self.hyper_b_1 = tf.keras.layers.Dense(embed_dim)

        # --- HyperNet for Layer 2 Weights ---
        self.hyper_w_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(hypernet_embed_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])

        self.hyper_b_2 = tf.keras.Sequential([
            tf.keras.layers.Dense(hypernet_embed_dim, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    @tf.function(reduce_retracing=True)
    def call(self, agent_qs, state):
        batch_size = tf.shape(agent_qs)[0]

        q_values = tf.reshape(agent_qs, (batch_size, 1, self.n_agents))

        w1 = tf.abs(self.hyper_w_1(state))
        w1 = tf.reshape(w1, (batch_size, self.n_agents, self.embed_dim))
        b1 = tf.reshape(self.hyper_b_1(state), (batch_size, 1, self.embed_dim))

        hidden = tf.nn.elu(tf.matmul(q_values, w1) + b1)

        w2 = self.hyper_w_2(state)
        w2 = tf.reshape(tf.abs(w2), (-1, self.embed_dim, 1))
        
        b2 = tf.reshape(self.hyper_b_2(state), (-1, 1, 1))

        q_tot = tf.matmul(hidden, w2) + b2
        q_tot = tf.reshape(q_tot, (batch_size, -1))

        return q_tot

def epsilon_greedy(q, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(0, N_ACTIONS)
    else:
        return int(np.argmax(q))

def get_global_state(env_observations):
    return np.concatenate(env_observations)

D_MAX = 50000
D_obs = np.zeros((D_MAX, N_AGENTS, N_OBS_DIM), dtype=np.float32)
D_actions = np.zeros((D_MAX, N_AGENTS), dtype=np.int32)
D_rewards = np.zeros((D_MAX, N_AGENTS), dtype=np.float32)
D_next_obs = np.zeros((D_MAX, N_AGENTS, N_OBS_DIM), dtype=np.float32)
D_states = np.zeros((D_MAX, N_AGENTS * N_OBS_DIM), dtype=np.float32)
D_next_states = np.zeros((D_MAX, N_AGENTS * N_OBS_DIM), dtype=np.float32)
D_dones = np.zeros((D_MAX,), dtype=np.float32)
D_prev_act = np.zeros((D_MAX, N_AGENTS), dtype=np.int32)
D_size = 0
D_ptr = 0

# Instantiate models
agent_network = QMixAgent(HIDDEN_DIM, N_ACTIONS)
mixer_network = QMixer(N_AGENTS, state_dim=(N_OBS_DIM * N_AGENTS), embed_dim=64, hypernet_embed_dim=64)

# Create dummy inputs to initialize weights
dummy_agent_input = tf.zeros((1, N_FEATURES))
_ = agent_network(dummy_agent_input)

dummy_mixer_q = tf.zeros((1, N_AGENTS))
dummy_mixer_state = tf.zeros((1, N_OBS_DIM * N_AGENTS))
_ = mixer_network(dummy_mixer_q, dummy_mixer_state)

target_agent = QMixAgent(HIDDEN_DIM, N_ACTIONS)
_ = target_agent(dummy_agent_input)
target_agent.set_weights(agent_network.get_weights())

target_mixer = QMixer(N_AGENTS, state_dim=(N_OBS_DIM * N_AGENTS), embed_dim=64, hypernet_embed_dim=64)
_ = target_mixer(dummy_mixer_q, dummy_mixer_state)
target_mixer.set_weights(mixer_network.get_weights())


@tf.function(reduce_retracing=True)
def train_step(flat_obs, b_actions, b_rewards, flat_next_obs, b_states, b_next_states, b_dones):
    with tf.GradientTape() as tape:
        all_q_vals = agent_network(flat_obs)
        all_q_vals = tf.reshape(all_q_vals, (BATCH_SIZE, N_AGENTS, N_ACTIONS))

        choose_action_q_vals = tf.squeeze(tf.gather(all_q_vals, b_actions, batch_dims=2), -1)
        q_tot = mixer_network(choose_action_q_vals, b_states)

        next_q_vals_online = agent_network(flat_next_obs)
        next_q_vals_online = tf.reshape(next_q_vals_online, (BATCH_SIZE, N_AGENTS, N_ACTIONS))
        next_actions_online = tf.expand_dims(tf.argmax(next_q_vals_online, axis=2, output_type=tf.int32), -1)

        target_q_vals = target_agent(flat_next_obs)
        target_q_vals = tf.reshape(target_q_vals, (BATCH_SIZE, N_AGENTS, N_ACTIONS))
        target_max_q_vals = tf.squeeze(tf.gather(target_q_vals, next_actions_online, batch_dims=2), -1)

        target_q_tot = target_mixer(target_max_q_vals, b_next_states)
        expected_q_tot = tf.stop_gradient(b_rewards + GAMMA * (1 - b_dones) * target_q_tot)

        loss = tf.keras.losses.MSE(expected_q_tot, q_tot)
        loss = tf.reduce_mean(loss)
    
    vars_to_train = agent_network.trainable_variables + mixer_network.trainable_variables
    grads = tape.gradient(loss, vars_to_train)
    grads, _ = tf.clip_by_global_norm(grads, 10.0)
    optimizer.apply_gradients(zip(grads, vars_to_train))

# lr_critic equivalent for QMIX value net
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)

# Metrics tracking
training_rewards = []
training_deliveries = []

epsilon = epsilon_start

import time
start_time = time.time()

for episode in range(NUM_EPISODES):
    observations = env.reset()
    global_state = get_global_state(observations)

    previous_actions = np.zeros(N_AGENTS)
    
    episode_reward = 0
    episode_deliveries = 0
    
    # Epsilon decay
    if episode < epsilon_decay_steps:
        epsilon = epsilon_start - episode * ((epsilon_start - epsilon_end) / epsilon_decay_steps)
    else:
        epsilon = epsilon_end

    for t in range(MAX_STEPS):
        current_actions = []

        # --- DECENTRALIZED EXECUTION (BATCHED) ---
        agent_id_oh = EYE_AGENTS
        prev_act_oh = EYE_ACTIONS[np.array(previous_actions, dtype=np.int32)]
        obs_array = np.array(observations, dtype=np.float32)

        combined_input = np.concatenate([agent_id_oh, obs_array, prev_act_oh], axis=1)
        q_values_batch = agent_network(combined_input).numpy()

        for i in range(N_AGENTS):
            action_i = epsilon_greedy(q_values_batch[i], epsilon)
            current_actions.append(action_i)

        # --- ENVIRONMENT STEP ---
        next_observations, rewards, dones, info = env.step(current_actions)
        next_global_state = get_global_state(next_observations)
        
        episode_reward += sum(rewards)
        # Note: MAPPO dense shaping outputs high values. Using > 0.9 correctly isolates 'Delivery Bonus' 
        episode_deliveries += sum(1 for r in rewards if r > 0.9)

        # --- STORE EXPERIENCE ---
        D_obs[D_ptr] = observations
        D_actions[D_ptr] = current_actions
        D_rewards[D_ptr] = rewards
        D_next_obs[D_ptr] = next_observations
        D_states[D_ptr] = global_state
        D_next_states[D_ptr] = next_global_state
        D_dones[D_ptr] = float(any(dones))
        D_prev_act[D_ptr] = previous_actions

        D_ptr = (D_ptr + 1) % D_MAX
        D_size = min(D_size + 1, D_MAX)

        observations = next_observations
        global_state = next_global_state
        previous_actions = current_actions
        
        if all(dones):
            break

    training_rewards.append(episode_reward)
    training_deliveries.append(episode_deliveries)
    
    # --- CENTRALIZED TRAINING ---
    # Throttle training to prevent severe overfitting to early state penalties and stabilize target Q-learning 
    if D_size >= MIN_REPLAY_SIZE and t % 10 == 0:
        # np.random.randint is O(1) compared to np.random.choice which allocates arrays taking O(N) 
        batch_indices = np.random.randint(0, D_size, size=BATCH_SIZE)
        
        b_obs = D_obs[batch_indices]
        b_actions = np.expand_dims(D_actions[batch_indices], -1) 
        b_rewards = np.expand_dims(np.sum(D_rewards[batch_indices], axis=1), -1)
        b_next_obs = D_next_obs[batch_indices]
        b_states = D_states[batch_indices]
        b_next_states = D_next_states[batch_indices]
        b_dones = np.expand_dims(D_dones[batch_indices], -1)
        b_prev_act = D_prev_act[batch_indices]
        
        b_agent_id_oh = np.broadcast_to(np.expand_dims(EYE_AGENTS, 0), (BATCH_SIZE, N_AGENTS, N_AGENTS))
        b_prev_act_oh = EYE_ACTIONS[b_prev_act]
        
        b_inputs = np.concatenate([b_agent_id_oh, b_obs, b_prev_act_oh], axis=-1)
        flat_obs = b_inputs.reshape((-1, N_FEATURES))
        
        b_act_oh = EYE_ACTIONS[np.squeeze(b_actions, axis=-1)]
        b_next_inputs = np.concatenate([b_agent_id_oh, b_next_obs, b_act_oh], axis=-1)
        flat_next_obs = b_next_inputs.reshape((-1, N_FEATURES))

        b_actions_tensor = tf.convert_to_tensor(b_actions, dtype=tf.int32)
        train_step(tf.convert_to_tensor(flat_obs, dtype=tf.float32), 
                   b_actions_tensor, 
                   tf.convert_to_tensor(b_rewards, dtype=tf.float32), 
                   tf.convert_to_tensor(flat_next_obs, dtype=tf.float32), 
                   tf.convert_to_tensor(b_states, dtype=tf.float32), 
                   tf.convert_to_tensor(b_next_states, dtype=tf.float32), 
                   tf.convert_to_tensor(b_dones, dtype=tf.float32))

    if episode % TARGET_UPDATE_INTERVAL == 0:
        target_agent.set_weights(agent_network.get_weights())
        target_mixer.set_weights(mixer_network.get_weights())
        
    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f} | Deliveries: {training_deliveries[-1]} | Time: {time.time()-start_time:.1f}s")
        
    if (episode + 1) % 100000 == 0:
        os.makedirs("results/checkpoints", exist_ok=True)
        agent_network.save_weights(f"results/checkpoints/qmix_tf_agent_ep{episode+1}.weights.h5")
        mixer_network.save_weights(f"results/checkpoints/qmix_tf_mixer_ep{episode+1}.weights.h5")
        print(f"--- Checkpoint saved at Episode {episode+1} ---")

os.makedirs("results/checkpoints", exist_ok=True)
agent_network.save_weights("results/checkpoints/qmix_tf_agent.weights.h5")
mixer_network.save_weights("results/checkpoints/qmix_tf_mixer.weights.h5")
print("Training finished! Weights saved.")

# ==========================================
# Evaluation vs MAPPO & Random Baseline
# ==========================================
import json
import os
import csv

EVAL_EPISODES = 300

print(f"Evaluating QMIX for {EVAL_EPISODES} episodes...")

q_r = []
q_d = []

for ep in range(EVAL_EPISODES):
    observations = env.reset()
    previous_actions = np.zeros(N_AGENTS)
    episode_reward = 0
    episode_deliveries = 0
    
    for t in range(MAX_STEPS):
        current_actions = []
        agent_id_oh = EYE_AGENTS
        prev_act_oh = EYE_ACTIONS[np.array(previous_actions, dtype=np.int32)]
        obs_array = np.array(observations, dtype=np.float32)

        combined_input = np.concatenate([agent_id_oh, obs_array, prev_act_oh], axis=1)
        q_values_batch = agent_network(combined_input).numpy()

        for i in range(N_AGENTS):
            # greedy evaluation
            action_i = int(np.argmax(q_values_batch[i]))
            current_actions.append(action_i)
            
        observations, rewards, dones, info = env.step(current_actions)
        previous_actions = current_actions
        episode_reward += sum(rewards)
        episode_deliveries += sum(1 for r in rewards if r > 0.9) # Dense reward shape
        
        if all(dones):
            break
            
    q_r.append(episode_reward)
    q_d.append(episode_deliveries)
    
qmix_mean_reward = np.mean(q_r)
print(f"QMIX Mean Reward: {qmix_mean_reward:.4f}")

def load_baseline(filepath):
    r_list, d_list = [], []
    try:
        with open(filepath) as f:
            data = json.load(f)
        if "episodes" in data:
            r_list = [ep["team_total_reward"] for ep in data["episodes"]][:EVAL_EPISODES]
            # Baselines unshaped rewards roughly equal discrete deliveries
            d_list = [ep["team_total_reward"] for ep in data["episodes"]][:EVAL_EPISODES]
        else:
            r_list = data.get("_rewards", [])[:EVAL_EPISODES]
            d_list = data.get("_deliveries", [])[:EVAL_EPISODES]
    except Exception:
        pass
    return r_list, d_list

# 1. Load MAPPO Baseline
m_r, m_d = load_baseline("results/logs/trained_policy_rewards.json")
if not m_r: print("Warning: MAPPO baseline failed to load.")

# 2. Load Random Baseline
r_r, r_d = load_baseline("results/logs/random_baseline_rewards.json")
if not r_r: print("Warning: Random baseline failed to load.")

# 3. Load Greedy Baseline
g_r, g_d = load_baseline("results/logs/greedy_baseline_rewards.json")
if not g_r: print("Warning: Greedy baseline failed to load.")

os.makedirs("results/logs", exist_ok=True)

# Save JSON
log_data = {"_rewards": q_r, "_deliveries": q_d, "mean_reward": float(qmix_mean_reward)}
with open("results/logs/qmix_reward.json", "w") as f:
    json.dump(log_data, f, indent=2)

# Save CSV
with open("results/logs/qmix_reward.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Reward", "Deliveries"])
    for ep, (r, d) in enumerate(zip(q_r, q_d)):
        writer.writerow([ep, float(r), int(d)])
        
print("Saved QMIX logs to results/logs/qmix_reward.json and .csv")

# Save Evaluation Summary to CSV
summary_csv_path = "results/logs/qmix_evaluation_summary.csv"
with open(summary_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Policy", "Mean Reward", "Std Reward", "Avg Deliveries", "Max Deliveries", "Delivery Rate (%)", "Positive Reward Rate (%)"])
    
    q_d_any = sum(1 for d in q_d if d > 0)
    q_pos = sum(1 for r in q_r if r > 0.0)
    writer.writerow([
        "QMIX", 
        f"{np.mean(q_r):.4f}", 
        f"{np.std(q_r):.4f}", 
        f"{np.mean(q_d):.3f}", 
        f"{np.max(q_d)}", 
        f"{(q_d_any/EVAL_EPISODES)*100:.1f}", 
        f"{(q_pos/EVAL_EPISODES)*100:.1f}"
    ])
    
    if len(m_r) > 1 and len(m_d) > 1:
        m_d_any = sum(1 for d in m_d if d > 0)
        m_pos = sum(1 for r in m_r if r > 0.0)
        writer.writerow([
            "MAPPO", 
            f"{np.mean(m_r):.4f}", 
            f"{np.std(m_r):.4f}", 
            f"{np.mean(m_d):.3f}", 
            f"{np.max(m_d)}", 
            f"{(m_d_any/len(m_d))*100:.1f}", 
            f"{(m_pos/len(m_r))*100:.1f}"
        ])
        
    if len(r_r) > 1 and len(r_d) > 1:
        r_d_any = sum(1 for d in r_d if d > 0)
        r_pos = sum(1 for r in r_r if r > 0.0)
        writer.writerow([
            "Random", 
            f"{np.mean(r_r):.4f}", 
            f"{np.std(r_r):.4f}", 
            f"{np.mean(r_d):.3f}", 
            f"{np.max(r_d)}", 
            f"{(r_d_any/len(r_d))*100:.1f}", 
            f"{(r_pos/len(r_r))*100:.1f}"
        ])

    if len(g_r) > 1 and len(g_d) > 1:
        g_d_any = sum(1 for d in g_d if d > 0)
        g_pos = sum(1 for r in g_r if r > 0.0)
        writer.writerow([
            "Greedy", 
            f"{np.mean(g_r):.4f}", 
            f"{np.std(g_r):.4f}", 
            f"{np.mean(g_d):.3f}", 
            f"{np.max(g_d)}", 
            f"{(g_d_any/len(g_d))*100:.1f}", 
            f"{(g_pos/len(g_r))*100:.1f}"
        ])

print(f"Saved QMIX Evaluation Summary to {summary_csv_path}")

# Statistical comparison logic helper
try:
    from scipy.stats import ttest_ind
    with open("results/logs/significance_tests.txt", "w") as sig_f:
        sig_f.write("QMIX vs Baselines Statistical Significance Tests\n")
        sig_f.write("="*50 + "\n")
        
        if len(m_r) > 1:
            t_stat, p_val = ttest_ind(q_r, m_r, equal_var=False)
            res = f"--- QMIX vs MAPPO Statistical Significance ---\nT-statistic: {t_stat:.4f} | P-value: {p_val:.4f} (Signif: {'YES' if p_val < 0.05 else 'NO'})\n"
            print(res); sig_f.write(res + "\n")
        
        if len(r_r) > 1:
            t_stat2, p_val2 = ttest_ind(q_r, r_r, equal_var=False)
            res2 = f"--- QMIX vs Random Statistical Significance ---\nT-statistic: {t_stat2:.4f} | P-value: {p_val2:.4f} (Signif: {'YES' if p_val2 < 0.05 else 'NO'})\n"
            print(res2); sig_f.write(res2 + "\n")
            
        if len(g_r) > 1:
            t_stat3, p_val3 = ttest_ind(q_r, g_r, equal_var=False)
            res3 = f"--- QMIX vs Greedy Statistical Significance ---\nT-statistic: {t_stat3:.4f} | P-value: {p_val3:.4f} (Signif: {'YES' if p_val3 < 0.05 else 'NO'})\n"
            print(res3); sig_f.write(res3 + "\n")
            
    print("Saved Statistical tests to results/logs/significance_tests.txt")
except ImportError:
    pass


# ==========================================
# Tri-Model 6-Panel Visualization Dashboard
# ==========================================

import matplotlib.pyplot as plt

q_r = np.array(q_r)
r_r = np.array(r_r)
m_r = np.array(m_r) if len(m_r) > 0 else np.array([])
g_r = np.array(g_r) if len(g_r) > 0 else np.array([])
q_d = np.array(q_d)
r_d = np.array(r_d)
m_d = np.array(m_d) if len(m_d) > 0 else np.array([])
g_d = np.array(g_d) if len(g_d) > 0 else np.array([])

QMIX_COLOR = "#00BCD4"
MAPPO_COLOR = "#AB47BC"
GREEDY_COLOR = "#FF9800"
RAND_COLOR = "#E0E0E0"

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("QMIX (TF) vs MAPPO vs Random Baseline Evaluation", fontsize=16, fontweight="bold")

# Panel 1: Histogram
ax = axes[0, 0]
all_vals = [q_r, r_r]
if len(m_r) > 0: all_vals.append(m_r)
if len(g_r) > 0: all_vals.append(g_r)
bins = np.histogram_bin_edges(np.concatenate(all_vals), bins=20)
ax.hist(q_r, bins=bins, alpha=0.6, color=QMIX_COLOR, label=f"QMIX (μ={q_r.mean():.2f})")
if len(m_r) > 0:
    ax.hist(m_r, bins=bins, alpha=0.6, color=MAPPO_COLOR, label=f"MAPPO (μ={m_r.mean():.2f})")
if len(g_r) > 0:
    ax.hist(g_r, bins=bins, alpha=0.6, color=GREEDY_COLOR, label=f"Greedy (μ={g_r.mean():.2f})")
ax.hist(r_r, bins=bins, alpha=0.6, color=RAND_COLOR, label=f"Random (μ={r_r.mean():.2f})")
ax.axvline(q_r.mean(), color=QMIX_COLOR, linewidth=2, linestyle="--")
if len(m_r) > 0:
    ax.axvline(m_r.mean(), color=MAPPO_COLOR, linewidth=2, linestyle="--")
if len(g_r) > 0:
    ax.axvline(g_r.mean(), color=GREEDY_COLOR, linewidth=2, linestyle="--")
ax.axvline(r_r.mean(), color="grey", linewidth=2, linestyle="--")
ax.set_title("Reward Distribution")
ax.legend()
ax.grid(alpha=0.3)

# Panel 2: Box Plot
ax = axes[0, 1]
bp_data = [q_r]
bp_labels = ["QMIX"]
bp_colors = [QMIX_COLOR]
if len(m_r) > 0:
    bp_data.append(m_r); bp_labels.append("MAPPO"); bp_colors.append(MAPPO_COLOR)
if len(g_r) > 0:
    bp_data.append(g_r); bp_labels.append("Greedy"); bp_colors.append(GREEDY_COLOR)
bp_data.append(r_r); bp_labels.append("Random"); bp_colors.append(RAND_COLOR)

bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True)
for patch, color in zip(bp["boxes"], bp_colors):
    patch.set_facecolor(color)
ax.set_title("Reward Box Plot")
ax.grid(alpha=0.3)

# Panel 3: Mean Deliveries
ax = axes[0, 2]
labels, means, stds, colors = ["QMIX"], [q_d.mean()], [q_d.std()], [QMIX_COLOR]
if len(m_d) > 0:
    labels.append("MAPPO"); means.append(m_d.mean()); stds.append(m_d.std()); colors.append(MAPPO_COLOR)
if len(g_d) > 0:
    labels.append("Greedy"); means.append(g_d.mean()); stds.append(g_d.std()); colors.append(GREEDY_COLOR)
labels.append("Random"); means.append(r_d.mean()); stds.append(r_d.std()); colors.append(RAND_COLOR)

bars = ax.bar(labels, means, color=colors, alpha=0.8, yerr=stds, capsize=8)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, m + max(stds)*0.05, f"{m:.2f}", ha="center", va="bottom")
ax.set_title("Mean Deliveries per Episode")
ax.grid(alpha=0.3)

# Panel 4: Training Curve Comparison
ax = axes[1, 0]
window = max(100, NUM_EPISODES // 100)
if len(training_rewards) >= window:
    step_size = max(1, window//2)
    smoothed_qmix = [np.mean(training_rewards[i:i+window]) for i in range(0, len(training_rewards)-window+1, step_size)]
    x_q = np.arange(len(smoothed_qmix)) * step_size
    ax.plot(x_q, smoothed_qmix, color=QMIX_COLOR, linewidth=2, label="QMIX Running Reward")
ax.axhline(r_r.mean(), color=RAND_COLOR, linestyle="--", label="Random Mean")
if len(m_r) > 0:
    ax.axhline(m_r.mean(), color=MAPPO_COLOR, linestyle="-.", label="MAPPO Mean (Eval)")
if len(g_r) > 0:
    ax.axhline(g_r.mean(), color=GREEDY_COLOR, linestyle=":", label="Greedy Mean (Eval)")

# Check if MAPPO Eval Curve can be plotted
mappo_curve_path = "results/logs/mappo_eval_curve.json"
try:
    with open(mappo_curve_path) as f:
        m_curve = json.load(f)
        m_ts = [e["timestep"] / 500.0 for e in m_curve] # Approximate to episodes
        m_rew = [e["eval_mean_reward"] for e in m_curve]
        ax.plot(m_ts, m_rew, color=MAPPO_COLOR, linewidth=1.5, linestyle="--", label="MAPPO (approx eps)")
except (FileNotFoundError, KeyError):
    pass

ax.set_title("Estimated Training Curves overlay")
ax.legend()
ax.grid(alpha=0.3)

# Panel 5: Cumulative Deliveries
ax = axes[1, 1]
ax.plot(np.cumsum(q_d), color=QMIX_COLOR, linewidth=2, label="QMIX (Eval)")
if len(m_d) > 0:
    ax.plot(np.cumsum(m_d), color=MAPPO_COLOR, linewidth=2, label="MAPPO (Eval)")
if len(g_d) > 0:
    ax.plot(np.cumsum(g_d), color=GREEDY_COLOR, linewidth=2, label="Greedy (Eval)")
ax.plot(np.cumsum(r_d), color=RAND_COLOR, linewidth=2, label="Random (Eval)")

# Add cumulative training deliveries scaled to evaluate growth dynamically
if len(training_deliveries) > 0:
    # Scale x-axis from 0 to EVAL_EPISODES to fit on the same graph visually
    x_train = np.linspace(0, EVAL_EPISODES, len(training_deliveries))
    ax.plot(x_train, np.cumsum(training_deliveries), color="blue", alpha=0.5, linestyle=":", linewidth=2, label="QMIX (Training Scaled)")

ax.set_title("Cumulative Deliveries")
ax.legend()
ax.grid(alpha=0.3)

# Panel 6: Positive Rate (Delivery Success)
ax = axes[1, 2]
eval_window = 50
# Track positive delivery rate (d > 0), because base rewards are structurally negative 
q_pos = [np.mean([1 if d > 0 else 0 for d in q_d[i:i+eval_window]]) for i in range(0, len(q_d)-eval_window+1, eval_window//2)]
r_pos = [np.mean([1 if d > 0 else 0 for d in r_d[i:i+eval_window]]) for i in range(0, len(r_d)-eval_window+1, eval_window//2)]
x_q_pos = np.arange(len(q_pos)) * (eval_window//2)
ax.plot(x_q_pos, q_pos, color=QMIX_COLOR, linewidth=2, label="QMIX")
if len(m_d) > 0:
    m_pos = [np.mean([1 if d > 0 else 0 for d in m_d[i:i+eval_window]]) for i in range(0, len(m_d)-eval_window+1, eval_window//2)]
    ax.plot(x_q_pos, m_pos, color=MAPPO_COLOR, linewidth=2, label="MAPPO")
if len(g_d) > 0:
    g_pos_d = [np.mean([1 if d > 0 else 0 for d in g_d[i:i+eval_window]]) for i in range(0, len(g_d)-eval_window+1, eval_window//2)]
    ax.plot(x_q_pos, g_pos_d, color=GREEDY_COLOR, linewidth=2, label="Greedy")
ax.plot(x_q_pos, r_pos, color=RAND_COLOR, linewidth=2, label="Random")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Delivery Success Rate (Rolling 50-ep)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
import os
os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/qmix_comparison_dashboard.png", dpi=150)
plt.show()
