# Deep Q-Network (DQN) for CartPole-v1

This project implements the Deep Q-Network (DQN) algorithm to train an agent to solve the CartPole-v1 environment. The implementation uses a standard **Multi-Layer Perceptron (MLP)** as the function approximator.

To stabilize training, this implementation uses two key techniques:

1.  **Experience Replay:** The agent stores its experiences (state, action, reward, next_state) in a replay buffer and samples random mini-batches from this buffer to perform updates, which breaks the correlation between consecutive samples.
2.  **Fixed Target Network:** A separate "target" network is used to generate the target Q-values for the loss calculation. This target network's weights are frozen for several steps and only periodically updated to match the main network, which helps prevent the target values from chasing the changing predictions.

== Requirements ==
To run this code, you need Python 3 and the following libraries:

* torch
* gymnasium
* numpy
* matplotlib

You can install them using pip:
```sh
pip install torch gymnasium numpy matplotlib
```

== How to Run the Code ==
The project is divided into two Python scripts:

* `dqn_agent_mlp_pytorch.py`: Defines the `DQNAgent` class, the `QNetwork` (MLP), and the `ReplayBuffer`.
* `train_cartpole_pytorch.py`: Contains the main training loop, environment interaction, and plotting logic.

After installing the requirements, you can run the experiment from your terminal. The script will train the model, print the final rollout results, and save the plots.
```sh
python train_cartpole_pytorch.py
```

== Algorithm Pseudocode ==
The following pseudocode outlines the DQN algorithm implemented in the scripts.

**Algorithm: Deep Q-Learning (DQN) with Experience Replay & Target Network**

1.  Initialize two identical neural networks:
    * `main_network` (weights $\theta$)
    * `target_network` (weights $\theta^-$ = $\theta$)
2.  Initialize the `replay_buffer` with a fixed capacity.
3.  Initialize exploration rate `epsilon` (e.g., to 1.0).
4.  **Loop for** a fixed number of episodes:
5.  Reset the environment to get the initial `state`.
6.  **Loop for** each step in the episode:
7.  **Choose Action (Epsilon-Greedy):**
    * With probability `epsilon`, select a `random_action`.
    * Otherwise, select `action` = $\text{argmax}_{a} Q_{\text{main}}(state, a; \theta)$.
8.  Execute `action` in the environment.
9.  Observe the `reward`, `next_state`, and `done` flag.
10. **Store Experience:**
    * Save the transition (`state`, `action`, `reward`, `next_state`, `done`) in the `replay_buffer`.
11. Set `state` = `next_state`.
12. **Train Main Network:**
    * If `replay_buffer` has enough samples:
    * Sample a random mini-batch of transitions from `replay_buffer`.
    * For each transition in the batch:
        * Calculate the target Q-value:
            * If `done`, `y_target` = `reward`.
            * If not `done`, `y_target` = `reward` + $\gamma \cdot \max_{a'} Q_{\text{target}}(next\_state, a'; \theta^-)$.
    * Calculate the current Q-value: `y_predicted` = $Q_{\text{main}}(state, action; \theta)$.
    * Calculate the loss (e.g., MSE): $L = (y_{\text{target}} - y_{\text{predicted}})^2$.
    * Perform a gradient descent step on $L$ to update the `main_network` weights $\theta$.
13. **Update Target Network:**
    * Every `C` steps, copy the `main_network` weights to the `target_network`: $\theta^- = \theta$.
14. Decay `epsilon` (e.g., `epsilon = epsilon * decay_rate`).
15. If `done`, break to the next episode.





# Deep Q-Network (DQN) for MsPacman-v0

This project implements the Deep Q-Network (DQN) algorithm to train an agent to play the Atari game Ms. Pac-Man (MsPacman-v0) directly from pixel input. The implementation uses a **Convolutional Neural Network (CNN)** as the function approximator, following the architecture described in the Mnih et al. (2015) paper.

Key features of this implementation include:

1.  **Image Preprocessing:** Game frames are converted to grayscale, downsampled, and cropped to 88x80 to reduce the state space.
2.  **Frame Stacking:** Four consecutive frames are stacked together to create a single state representation (88x80x4), allowing the agent to infer motion.
3.  **Experience Replay:** The agent stores (state, action, reward, next_state) transitions in a large replay buffer and samples mini-batches from it for training.
4.  **Fixed Target Network:** A separate "target" CNN is used to generate stable target Q-values for the loss calculation.

== Requirements ==
To run this code, you need Python 3 and the following libraries:

* torch
* gymnasium
* numpy
* matplotlib
* ale-py
* shimmy

You can install them using pip (this will include the Atari ROMs):
```sh
pip install torch gymnasium numpy matplotlib "gymnasium[atari, accept-rom-license]" shimmy
```

== How to Run the Code ==
The project is divided into two Python scripts:

* `dqn_agent_cnn_pytorch.py`: Defines the `AtariDQNAgent` class, the `QNetwork` (CNN), and the `ReplayBuffer`.
* `train_mspacman_pytorch.py`: Contains the main training loop, environment setup (including preprocessing and frame stacking), and plotting logic.

After installing the requirements, you can run the experiment from your terminal. The script will train the model, print the final rollout results, and save the plots.
```sh
python train_mspacman_pytorch.py
```

== Algorithm Pseudocode ==
The following pseudocode outlines the DQN algorithm implemented in the scripts. The core logic is identical to the CartPole example, but the state now represents a stack of preprocessed image frames and the networks are CNNs.

**Algorithm: Deep Q-Learning (DQN) with Experience Replay & Target Network**

1.  Initialize two identical CNNs:
    * `main_network` (weights $\theta$)
    * `target_network` (weights $\theta^-$ = $\theta$)
2.  Initialize the `replay_buffer` with a large capacity (e.g., 100,000).
3.  Initialize `epsilon` (e.g., to 1.0) and `total_steps` = 0.
4.  **Loop for** a fixed number of episodes:
5.  Reset the environment. Get the first frame, preprocess it, and stack it 4 times to create the initial `state`.
6.  **Loop for** each step in the episode:
7.  **Choose Action (Epsilon-Greedy):**
    * Linearly anneal `epsilon` from 1.0 to 0.1 based on `total_steps`.
    * With probability `epsilon`, select a `random_action`.
    * Otherwise, select `action` = $\text{argmax}_{a} Q_{\text{main}}(state, a; \theta)$.
8.  Execute `action` in the environment.
9.  Observe the `reward` and the `next_frame`.
10. Preprocess `next_frame` and append it to the frame stack to get `next_state`.
11. **Store Experience:**
    * Save the transition (`state`, `action`, `reward`, `next_state`, `done`) in the `replay_buffer`.
12. Set `state` = `next_state` and increment `total_steps`.
13. **Train Main Network:**
    * If `replay_buffer` size > `MIN_REPLAY_SIZE`:
    * Sample a random mini-batch of transitions from `replay_buffer`.
    * For each transition in the batch:
        * Calculate the target Q-value:
            * If `done`, `y_target` = `reward`.
            * If not `done`, `y_target` = `reward` + $\gamma \cdot \max_{a'} Q_{\text{target}}(next\_state, a'; \theta^-)$.
    * Calculate the current Q-value: `y_predicted` = $Q_{\text{main}}(state, action; \theta)$.
    * Calculate the loss (e.g., MSE): $L = (y_{\text{target}} - y_{\text{predicted}})^2$.
    * Perform a gradient descent step on $L$ to update the `main_network` weights $\theta$.
14. **Update Target Network:**
    * Every `TARGET_UPDATE_FREQ` steps, copy the `main_network` weights to the `target_network`: $\theta^- = \theta$.
15. If `done`, break to the next episode.
