import re

with open('qmix_tensorflow.py', 'r') as f:
    content = f.read()

# 1. Add EYE_AGENTS
content = content.replace("N_FEATURES = N_AGENTS + N_OBS_DIM + N_ACTIONS\n",
    "N_FEATURES = N_AGENTS + N_OBS_DIM + N_ACTIONS\n\n"
    "EYE_AGENTS = np.eye(N_AGENTS, dtype=np.float32)\n"
    "EYE_ACTIONS = np.eye(N_ACTIONS, dtype=np.float32)\n")

# 2. Delete get_agent_Q
content = re.sub(r'def get_agent_Q\(.*?return q_values\n', '', content, flags=re.DOTALL)

# 3. Replay Buffer pre-allocation
old_buffer = """D = []
D_ptr = 0
D_MAX = 50000"""

new_buffer = """D_MAX = 50000
D_obs = np.zeros((D_MAX, N_AGENTS, N_OBS_DIM), dtype=np.float32)
D_actions = np.zeros((D_MAX, N_AGENTS), dtype=np.int32)
D_rewards = np.zeros((D_MAX, N_AGENTS), dtype=np.float32)
D_next_obs = np.zeros((D_MAX, N_AGENTS, N_OBS_DIM), dtype=np.float32)
D_states = np.zeros((D_MAX, N_AGENTS * N_OBS_DIM), dtype=np.float32)
D_next_states = np.zeros((D_MAX, N_AGENTS * N_OBS_DIM), dtype=np.float32)
D_dones = np.zeros((D_MAX,), dtype=np.float32)
D_prev_act = np.zeros((D_MAX, N_AGENTS), dtype=np.int32)
D_size = 0
D_ptr = 0"""

content = content.replace(old_buffer, new_buffer)

# 4. Add train_step function
add_train_step_before = """# lr_critic equivalent for QMIX value net
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)"""

train_step_code = """
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

"""

content = content.replace(add_train_step_before, train_step_code + add_train_step_before)

# 5. Batched execution in rollout
old_rollout = """        # --- DECENTRALIZED EXECUTION ---
        for i in range(N_AGENTS):
            obs_i = observations[i]
            prev_act_i = previous_actions[i]

            q_values = get_agent_Q(agent_network, i, obs_i, prev_act_i)

            action_i = epsilon_greedy(q_values.numpy()[0], epsilon)
            current_actions.append(action_i)"""

new_rollout = """        # --- DECENTRALIZED EXECUTION (BATCHED) ---
        agent_id_oh = EYE_AGENTS
        prev_act_oh = EYE_ACTIONS[np.array(previous_actions, dtype=np.int32)]
        obs_array = np.array(observations, dtype=np.float32)

        combined_input = np.concatenate([agent_id_oh, obs_array, prev_act_oh], axis=1)
        q_values_batch = agent_network(combined_input).numpy()

        for i in range(N_AGENTS):
            action_i = epsilon_greedy(q_values_batch[i], epsilon)
            current_actions.append(action_i)"""

content = content.replace(old_rollout, new_rollout)

# 6. Store Experience
old_store = """        # --- STORE EXPERIENCE ---
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
            D_ptr = (D_ptr + 1) % D_MAX"""

new_store = """        # --- STORE EXPERIENCE ---
        D_obs[D_ptr] = observations
        D_actions[D_ptr] = current_actions
        D_rewards[D_ptr] = rewards
        D_next_obs[D_ptr] = next_observations
        D_states[D_ptr] = global_state
        D_next_states[D_ptr] = next_global_state
        D_dones[D_ptr] = float(any(dones))
        D_prev_act[D_ptr] = previous_actions

        D_ptr = (D_ptr + 1) % D_MAX
        D_size = min(D_size + 1, D_MAX)"""
        
content = content.replace(old_store, new_store)

# 7. Centralized Training replace
old_train = """    # --- CENTRALIZED TRAINING ---
    # Throttle training to prevent severe overfitting to early state penalties and stabilize target Q-learning 
    if len(D) >= MIN_REPLAY_SIZE and t % 10 == 0:
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
        optimizer.apply_gradients(zip(grads, vars_to_train))"""

new_train = """    # --- CENTRALIZED TRAINING ---
    # Throttle training to prevent severe overfitting to early state penalties and stabilize target Q-learning 
    if D_size >= MIN_REPLAY_SIZE and t % 10 == 0:
        batch_indices = np.random.choice(D_size, BATCH_SIZE, replace=False)
        
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
                   tf.convert_to_tensor(b_dones, dtype=tf.float32))"""

content = content.replace(old_train, new_train)

# 8. Batched execution in Eval loop
old_eval_rollout = """        for i in range(N_AGENTS):
            obs_i = observations[i]
            prev_act_i = previous_actions[i]
            
            q_values = get_agent_Q(agent_network, i, obs_i, prev_act_i)
            # greedy evaluation
            action_i = int(np.argmax(q_values.numpy()[0]))
            current_actions.append(action_i)"""

new_eval_rollout = """        agent_id_oh = EYE_AGENTS
        prev_act_oh = EYE_ACTIONS[np.array(previous_actions, dtype=np.int32)]
        obs_array = np.array(observations, dtype=np.float32)

        combined_input = np.concatenate([agent_id_oh, obs_array, prev_act_oh], axis=1)
        q_values_batch = agent_network(combined_input).numpy()

        for i in range(N_AGENTS):
            # greedy evaluation
            action_i = int(np.argmax(q_values_batch[i]))
            current_actions.append(action_i)"""

content = content.replace(old_eval_rollout, new_eval_rollout)

with open('qmix_tensorflow.py', 'w') as f:
    f.write(content)
