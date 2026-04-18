MAPPO is an on-policy actor-critic algorithm that explores via action entropy, while QMIX is an off-policy value-based algorithm that relies entirely on ϵ-greedy exploration. The current setup forces QMIX to behave like MAPPO, which breaks its ability to learn.


1. The Critical Blunder: Epsilon Initialization

The Issue: 
For an off-policy Q-learning algorithm, starting with ϵ=0.2 is a fatal error. It means the agent is taking the argmax of its randomly initialized neural network 80% of the time from episode 1. It will immediately latch onto a meaningless, repetitive behavior (like walking into a wall or spinning in place) and never randomly stumble upon a successful sequence of actions (finding a shelf, picking it up, carrying it to a goal).

The Fix: QMIX must start with 100% random exploration and gradually decay.


2. Reward Shaping Overhaul (env_config.yaml)

The Issue: The current reward shape is heavily punitive. A step_penalty of -0.005 over 500 steps equals -2.5 per agent, or -5.0 for the team—which perfectly aligns with the -5 to -7 reward floor shown in the dashboard. Coupled with a low delivery_bonus (0.5), the agent quickly learns that "doing anything causes pain" and simply gives up to minimize linger_penalty and bad_drop_penalty.

The Fix: Shift from a punitive structure to a dense, heavily positive reinforcement structure. Make the "cookie" so big that it outweighs the noise of exploration.


3. Update Frequency (Sample Inefficiency)

The Issue: The training loop updates the network every 4 episodes (episode % 4 == 0), running 8 total batches (4 epochs * 2 batches). This equals 8 updates per ~2000 environment steps. MAPPO can handle infrequent updates because it computes advantages over whole trajectories. QMIX needs far more frequent gradient steps to propagate the temporal difference (TD) error backward from the sparse delivery rewards to the start of the episode.

The Fix: Train every episode to increase the update-to-step ratio.


4. Architectural Limitation: The Missing Memory

The Issue: The Robot Warehouse (rware) environment is a highly Partially Observable Markov Decision Process (POMDP). Agents only see a small grid around them. Once they see a goal and turn around, they immediately "forget" where it is. The QMixAgent is a pure Multi-Layer Perceptron (MLP) with two Dense layers. It has no memory. The original QMIX paper uses a Recurrent Neural Network (RNN) specifically to solve this.

The Fix: While fixing the Epsilon and Rewards will get the agent learning something, it will eventually hit a ceiling without memory. Upgrade the QMixAgent to a DRQN (Deep Recurrent Q-Network).

##################################################################

1. Drastic Exploration Overhaul

The current epsilon_decay_steps covers 3,600 episodes (approx. 1.8M steps). Recent research on rware suggests that QMIX often requires 20M+ steps and much longer exploration periods to "find" the delivery reward.

    Extended Annealing: Increase the total training to at least 20,000+ episodes (approx. 10M steps) and anneal epsilon over the first 50-70% of that time.

    Action Masking: If the environment provides a mask of valid actions (e.g., don't try to pick up if no shelf is present), apply it in get_actions_tf. This prevents the agents from wasting millions of steps on impossible actions.
        * This prevents the agent from wasting exploration steps on impossible transitions and forces the gradients to only update the "reachable" action-value space, significantly reducing the complexity of the learning task.

2. Replay Buffer & Sampling Improvements

The current buffer size is 50,000 steps, which is relatively small for a multi-agent task.

    Increase Buffer Size: Move to 200,000 - 500,000 steps. Warehouse tasks require agents to remember long sequences of actions (moving to shelf → picking → moving to goal). A larger buffer ensures successful delivery trajectories stay in memory longer.

    Batch Size: Increase BATCH_SIZE from 128 to 256 or 512 to stabilize the gradient updates for the Mixer network.

3. Architecture & Training Tweaks

The "Monotonicity Constraint" in QMIX can sometimes be too restrictive for complex coordination.

    Optimizer: While you are using RMSprop, many state-of-the-art QMIX implementations favor Adam with a slightly higher learning rate (e.g., 5e-4) and weight decay specifically for the Mixer network to prevent it from collapsing too early into a simple summation.

    Double Q-Learning: Ensure your train_step effectively uses the Target Agent to select actions for the next state but the Target Mixer to evaluate them. This prevents the severe overestimation common in MARL.

4. Environment & Reward Shaping

The env_config.yaml has reward_type: "individual".

    Switch to "global": QMIX is designed to factorize a global reward into individual utilities. By using individual rewards, it might be confusing the Mixer, which expects a single team-based signal Qtot​ to decompose.

    Reward Density: Since the success rate is currently 0% in many evaluations, the agents aren't seeing the delivery_bonus.

        Temporary Dense Rewards: Keep the move_toward_shelf and carry_toward_goal shaping active, but perhaps increase their weight slightly until the first successful delivery is recorded, then decay them to favor the sparse delivery_bonus.

        Delivery Bonus: Increase this to +10.0 or +20.0. It needs to be an order of magnitude larger than any intermediate reward so the Q-mixer can clearly identify it as the primary objective.

        Move Toward Shelf: Reduce this to +0.01. It should be just enough to break the "stay still" local optima, but low enough that the agent doesn't just "dance" near shelves to collect reward without picking them up.

         Carry Toward Goal: Keep this at +0.05. Once an agent has a shelf, this is the most critical guidance to ensure it doesn't wander aimlessly.
