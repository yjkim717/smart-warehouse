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
NUM_EPISODES = 5
MAX_STEPS = config['env'].get('max_steps', 500)
GAMMA = 0.99
BATCH_SIZE = 128
TARGET_UPDATE_INTERVAL = 200

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
        D.append(transitions)
        
        # Limit buffer roughly if memory bounds matter, dropping old samples (e.g. 500k transitions max)
        if len(D) > 50000:
            D.pop(0)

        observations = next_observations
        global_state = next_global_state
        previous_actions = current_actions
        
        if all(dones):
            break

    training_rewards.append(episode_reward)
    training_deliveries.append(episode_deliveries)
    
    # --- CENTRALIZED TRAINING ---
    if len(D) >= BATCH_SIZE:
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

            target_q_vals = target_agent(flat_next_obs)
            target_q_vals = tf.reshape(target_q_vals, (BATCH_SIZE, N_AGENTS, N_ACTIONS))
            target_max_q_vals = tf.reduce_max(target_q_vals, axis=2)

            target_q_tot = target_mixer(target_max_q_vals, b_next_states)
            expected_q_tot = b_rewards + GAMMA * (1 - b_dones) * target_q_tot

            loss = tf.keras.losses.MSE(expected_q_tot, q_tot)
            loss = tf.reduce_mean(loss)
        
        vars_to_train = agent_network.trainable_variables + mixer_network.trainable_variables
        grads = tape.gradient(loss, vars_to_train)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        optimizer.apply_gradients(zip(grads, vars_to_train))

    if episode % TARGET_UPDATE_INTERVAL == 0:
        target_agent.set_weights(agent_network.get_weights())
        target_mixer.set_weights(mixer_network.get_weights())
        
    if True:
        print(f"Episode: {episode+1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f} | Deliveries: {training_deliveries[-1]} | Time: {time.time()-start_time:.1f}s")

os.makedirs("results/checkpoints", exist_ok=True)
agent_network.save_weights("results/checkpoints/qmix_tf_agent.weights.h5")
mixer_network.save_weights("results/checkpoints/qmix_tf_mixer.weights.h5")
print("Training finished! Weights saved.")

# ==========================================
