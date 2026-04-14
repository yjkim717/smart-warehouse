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