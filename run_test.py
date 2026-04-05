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

# Sync with MAPPO Hyperparameters
HIDDEN_DIM = 128
MIXER_HIDDEN_DIM = 128
NUM_EPISODES = 100
MAX_STEPS = config['env'].get('max_steps', 500)
GAMMA = 0.99
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_INTERVAL = 2  # Update every 2 episodes (~1000 training steps, aligns with standard DQN)

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

def get_agent_Q(agent_network, agent_id, observation, prev_action):
    agent_id_one_hot = np.eye(N_AGENTS)[agent_id]
    action_one_hot = np.eye(N_ACTIONS)[int(prev_action)]

    combined_input = np.concatenate([agent_id_one_hot, observation, action_one_hot])
    input_tensor = tf.expand_dims(tf.convert_to_tensor(combined_input, dtype=tf.float32), 0)

    q_values = agent_network(input_tensor)

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

D = []
D_ptr = 0
D_MAX = 50000

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

        # --- DECENTRALIZED EXECUTION ---
        for i in range(N_AGENTS):
            obs_i = observations[i]
            prev_act_i = previous_actions[i]

            q_values = get_agent_Q(agent_network, i, obs_i, prev_act_i)

            action_i = epsilon_greedy(q_values.numpy()[0], epsilon)
            current_actions.append(action_i)

        # --- ENVIRONMENT STEP ---
        next_observations, rewards, dones, info = env.step(current_actions)
        next_global_state = get_global_state(next_observations)
        
        episode_reward += sum(rewards)
        # Note: MAPPO dense shaping outputs high values. Using > 0.9 correctly isolates 'Delivery Bonus' 
        episode_deliveries += sum(1 for r in rewards if r > 0.9)

        # --- STORE EXPERIENCE ---
        transitions = {
            "obs": observations,
            "actions": current_actions,
            "rewards": rewards,
            "next_obs": next_observations,
            "state": global_state,
            "next_state": next_global_state,
            "dones": dones,
            "prev_actions": list(previous_actions)
        }
        
        if len(D) < D_MAX:
            D.append(transitions)
        else:
            D[D_ptr] = transitions
            D_ptr = (D_ptr + 1) % D_MAX

        observations = next_observations
        global_state = next_global_state
        previous_actions = current_actions
        
        if all(dones):
            break

    training_rewards.append(episode_reward)
    training_deliveries.append(episode_deliveries)
    
    # --- CENTRALIZED TRAINING ---
    if len(D) >= MIN_REPLAY_SIZE:
        batch_indices = np.random.choice(len(D), BATCH_SIZE, replace=False)
        batch = [D[idx] for idx in batch_indices]

        b_obs = tf.convert_to_tensor([b['obs'] for b in batch], dtype=tf.float32)
        b_actions = tf.expand_dims(tf.convert_to_tensor([b['actions'] for b in batch], dtype=tf.int32), -1)
        b_rewards = tf.reduce_sum(tf.convert_to_tensor([b['rewards'] for b in batch], dtype=tf.float32), axis=1, keepdims=True)
        b_next_obs = tf.convert_to_tensor([b['next_obs'] for b in batch], dtype=tf.float32)
        b_states = tf.convert_to_tensor([b['state'] for b in batch], dtype=tf.float32)
        b_next_states = tf.convert_to_tensor([b['next_state'] for b in batch], dtype=tf.float32)
        
        b_dones = tf.expand_dims(tf.convert_to_tensor([any(b['dones']) for b in batch], dtype=tf.float32), -1)

        b_agent_ids = tf.broadcast_to(tf.expand_dims(tf.range(N_AGENTS), 0), (BATCH_SIZE, N_AGENTS))
        b_agent_id_oh = tf.cast(tf.one_hot(b_agent_ids, depth=N_AGENTS), tf.float32)

        b_prev_act = tf.convert_to_tensor(np.array([b['prev_actions'] for b in batch]), dtype=tf.int32)
        b_prev_act_oh = tf.cast(tf.one_hot(b_prev_act, depth=N_ACTIONS), tf.float32)
        
        b_inputs = tf.concat([b_agent_id_oh, b_obs, b_prev_act_oh], axis=-1)
        flat_obs = tf.reshape(b_inputs, (-1, N_FEATURES))

        b_act_oh = tf.cast(tf.one_hot(tf.squeeze(b_actions, -1), depth=N_ACTIONS), tf.float32)
        b_next_inputs = tf.concat([b_agent_id_oh, b_next_obs, b_act_oh], axis=-1)
        flat_next_obs = tf.reshape(b_next_inputs, (-1, N_FEATURES))

        with tf.GradientTape() as tape:
            all_q_vals = agent_network(flat_obs)
            all_q_vals = tf.reshape(all_q_vals, (BATCH_SIZE, N_AGENTS, N_ACTIONS))

            choose_action_q_vals = tf.squeeze(tf.gather(all_q_vals, b_actions, batch_dims=2), -1)
            q_tot = mixer_network(choose_action_q_vals, b_states)

            # Double Q-learning: get argmax from online network
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

    if episode % TARGET_UPDATE_INTERVAL == 0:
        target_agent.set_weights(agent_network.get_weights())
        target_mixer.set_weights(mixer_network.get_weights())
        
    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f} | Deliveries: {training_deliveries[-1]} | Time: {time.time()-start_time:.1f}s")

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

EVAL_EPISODES = 5

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
        for i in range(N_AGENTS):
            obs_i = observations[i]
            prev_act_i = previous_actions[i]
            
            q_values = get_agent_Q(agent_network, i, obs_i, prev_act_i)
            # greedy evaluation
            action_i = int(np.argmax(q_values.numpy()[0]))
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

# 1. Load MAPPO Baseline
mappo_res_path = "results/logs/trained_policy_rewards.json"
m_r = []
m_d = []
try:
    with open(mappo_res_path) as f:
        mappo_res = json.load(f)
    m_r = mappo_res["_rewards"]
    m_d = mappo_res["_deliveries"]
    print(f"Loaded MAPPO Baseline. Mean Reward: {np.mean(m_r):.4f}")
except (FileNotFoundError, KeyError):
    print(f"Warning: {mappo_res_path} not found. Skipping MAPPO.")

# 2. Load Random Baseline
random_res_path = "results/logs/random_baseline_rewards.json"
r_r = []
r_d = []
try:
    with open(random_res_path) as f:
        random_res = json.load(f)
    r_r = random_res["_rewards"]
    r_d = random_res["_deliveries"]
    print(f"Loaded Random Baseline. Mean Reward: {np.mean(r_r):.4f}")
except (FileNotFoundError, KeyError):
    print(f"Warning: {random_res_path} not found. Running ad-hoc random eval.")
    for ep in range(EVAL_EPISODES):
        env.reset()
        episode_reward = 0
        episode_deliveries = 0
        for t in range(MAX_STEPS):
            actions = [np.random.randint(0, N_ACTIONS) for _ in range(N_AGENTS)]
            _, rewards, dones, _ = env.step(actions)
            episode_reward += sum(rewards)
            episode_deliveries += sum(1 for r in rewards if r > 0.9)
            if all(dones):
                break
        r_r.append(episode_reward)
        r_d.append(episode_deliveries)

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

# Statistical comparison logic helper
try:
    from scipy.stats import ttest_ind
    if len(m_r) > 1:
        t_stat, p_val = ttest_ind(q_r, m_r, equal_var=False)
        print(f"--- QMIX vs MAPPO Statistical Significance ---")
        print(f"T-statistic: {t_stat:.4f} | P-value: {p_val:.4f}")
    t_stat2, p_val2 = ttest_ind(q_r, r_r, equal_var=False)
    print(f"--- QMIX vs Random Statistical Significance ---")
    print(f"T-statistic: {t_stat2:.4f} | P-value: {p_val2:.4f}")
except ImportError:
    pass

# ==========================================
# Tri-Model 6-Panel Visualization Dashboard
# ==========================================

import matplotlib.pyplot as plt

q_r = np.array(q_r)
r_r = np.array(r_r)
m_r = np.array(m_r) if len(m_r) > 0 else np.array([])
q_d = np.array(q_d)
r_d = np.array(r_d)
m_d = np.array(m_d) if len(m_d) > 0 else np.array([])

QMIX_COLOR = "#00BCD4"
MAPPO_COLOR = "#AB47BC"
RAND_COLOR = "#E0E0E0"

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("QMIX (TF) vs MAPPO vs Random Baseline Evaluation", fontsize=16, fontweight="bold")

# Panel 1: Histogram
ax = axes[0, 0]
all_vals = [q_r, r_r]
if len(m_r) > 0: all_vals.append(m_r)
bins = np.histogram_bin_edges(np.concatenate(all_vals), bins=20)
ax.hist(q_r, bins=bins, alpha=0.6, color=QMIX_COLOR, label=f"QMIX (μ={q_r.mean():.2f})")
if len(m_r) > 0:
    ax.hist(m_r, bins=bins, alpha=0.6, color=MAPPO_COLOR, label=f"MAPPO (μ={m_r.mean():.2f})")
ax.hist(r_r, bins=bins, alpha=0.6, color=RAND_COLOR, label=f"Random (μ={r_r.mean():.2f})")
ax.axvline(q_r.mean(), color=QMIX_COLOR, linewidth=2, linestyle="--")
if len(m_r) > 0:
    ax.axvline(m_r.mean(), color=MAPPO_COLOR, linewidth=2, linestyle="--")
ax.axvline(r_r.mean(), color="grey", linewidth=2, linestyle="--")
ax.set_title("Reward Distribution")
ax.legend()
ax.grid(alpha=0.3)

# Panel 2: Box Plot
ax = axes[0, 1]
if len(m_r) > 0:
    bp = ax.boxplot([q_r, m_r, r_r], labels=["QMIX", "MAPPO", "Random"], patch_artist=True)
    bp["boxes"][0].set_facecolor(QMIX_COLOR)
    bp["boxes"][1].set_facecolor(MAPPO_COLOR)
    bp["boxes"][2].set_facecolor(RAND_COLOR)
else:
    bp = ax.boxplot([q_r, r_r], labels=["QMIX", "Random"], patch_artist=True)
    bp["boxes"][0].set_facecolor(QMIX_COLOR)
    bp["boxes"][1].set_facecolor(RAND_COLOR)
ax.set_title("Reward Box Plot")
ax.grid(alpha=0.3)

# Panel 3: Mean Deliveries
ax = axes[0, 2]
if len(m_d) > 0:
    labels, means, stds, colors = ["QMIX", "MAPPO", "Random"], [q_d.mean(), m_d.mean(), r_d.mean()], [q_d.std(), m_d.std(), r_d.std()], [QMIX_COLOR, MAPPO_COLOR, RAND_COLOR]
else:
    labels, means, stds, colors = ["QMIX", "Random"], [q_d.mean(), r_d.mean()], [q_d.std(), r_d.std()], [QMIX_COLOR, RAND_COLOR]
bars = ax.bar(labels, means, color=colors, alpha=0.8, yerr=stds, capsize=8)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, m + max(stds)*0.05, f"{m:.2f}", ha="center", va="bottom")
ax.set_title("Mean Deliveries per Episode")
ax.grid(alpha=0.3)

# Panel 4: Training Curve Comparison
ax = axes[1, 0]
window = 100
if len(training_rewards) >= window:
    smoothed_qmix = [np.mean(training_rewards[i:i+window]) for i in range(0, len(training_rewards)-window+1, window//2)]
    x_q = np.arange(len(smoothed_qmix)) * (window//2)
    ax.plot(x_q, smoothed_qmix, color=QMIX_COLOR, linewidth=2, label="QMIX Running Reward")
ax.axhline(r_r.mean(), color=RAND_COLOR, linestyle="--", label="Random Mean")
if len(m_r) > 0:
    ax.axhline(m_r.mean(), color=MAPPO_COLOR, linestyle="-.", label="MAPPO Mean (Eval)")

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
ax.plot(np.cumsum(q_d), color=QMIX_COLOR, linewidth=2, label="QMIX")
if len(m_d) > 0:
    ax.plot(np.cumsum(m_d), color=MAPPO_COLOR, linewidth=2, label="MAPPO")
ax.plot(np.cumsum(r_d), color=RAND_COLOR, linewidth=2, label="Random")
ax.set_title("Cumulative Deliveries Over Evaluation Episodes")
ax.legend()
ax.grid(alpha=0.3)

# Panel 6: Positive Rate
ax = axes[1, 2]
eval_window = 50
q_pos = [np.mean([1 if r > 0 else 0 for r in q_r[i:i+eval_window]]) for i in range(0, len(q_r)-eval_window+1, eval_window//2)]
r_pos = [np.mean([1 if r > 0 else 0 for r in r_r[i:i+eval_window]]) for i in range(0, len(r_r)-eval_window+1, eval_window//2)]
x_q_pos = np.arange(len(q_pos)) * (eval_window//2)
ax.plot(x_q_pos, q_pos, color=QMIX_COLOR, label="QMIX")
if len(m_r) > 0:
    m_pos = [np.mean([1 if r > 0 else 0 for r in m_r[i:i+eval_window]]) for i in range(0, len(m_r)-eval_window+1, eval_window//2)]
    ax.plot(x_q_pos, m_pos, color=MAPPO_COLOR, label="MAPPO")
ax.plot(x_q_pos, r_pos, color=RAND_COLOR, label="Random")
ax.set_ylim(0, 1.05)
ax.set_title("Positive Rate (Rolling 50-ep)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
import os
os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/qmix_comparison_dashboard.png", dpi=150)
plt.show()
