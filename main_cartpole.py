import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from dqn_agent_mlp_pytorch import DQNAgent, device  
from collections import deque
import time
import torch 

ENV_NAME = "CartPole-v1"
DISCOUNT_FACTOR = 0.95
NUM_EPISODES = 2000 
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10 

env = gym.make(ENV_NAME)
state_shape = env.observation_space.shape
num_actions = env.action_space.n

agent = DQNAgent(state_shape, num_actions, DISCOUNT_FACTOR, lr=1e-3, replay_buffer_size=50000)

all_episode_rewards = []
all_max_q_values = []
rewards_window = deque(maxlen=100)
all_moving_avgs = []

start_time = time.time()

for episode in range(NUM_EPISODES):
    state, info = env.reset()
    episode_reward = 0
    max_q_this_episode = -float('inf')
    done = False
    
    while not done:
        action = agent.choose_action(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.replay_buffer.add(state, action, reward, next_state, done)
        
        loss = agent.train(BATCH_SIZE)
        
        state = next_state
        episode_reward += reward
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values_pred = agent.main_network(state_tensor).cpu().numpy()[0]
            max_q_this_episode = max(max_q_this_episode, np.max(q_values_pred))

    all_episode_rewards.append(episode_reward)
    rewards_window.append(episode_reward)
    all_moving_avgs.append(np.mean(rewards_window))
    all_max_q_values.append(max_q_this_episode)
    
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    agent.update_epsilon()
    

end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

rollout_rewards = []
for _ in range(500):
    state, info = env.reset()
    episode_reward = 0
    done = False
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent.main_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        if episode_reward >= 500: 
            break
            
    rollout_rewards.append(episode_reward)

mean_reward = np.mean(rollout_rewards)
std_reward = np.std(rollout_rewards)
print(f"Rollout Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


plt.figure(figsize=(12, 6))
plt.plot(all_max_q_values)
plt.title(f'{ENV_NAME} - Max Q-Value vs. Training Episodes (PyTorch)')
plt.xlabel('Training Episode')
plt.ylabel('Max Q-Value')
plt.grid(True)
plt.savefig(f"{ENV_NAME}_q_values_pytorch.png")
plt.close() 

plt.figure(figsize=(12, 6))
plt.plot(all_episode_rewards, label='Episode Reward', alpha=0.7)
plt.plot(all_moving_avgs, label='100-Episode Moving Average', color='red', linewidth=2)
plt.title(f'{ENV_NAME} - Episode Rewards vs. Training Episodes (PyTorch)')
plt.xlabel('Training Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.savefig(f"{ENV_NAME}_rewards_pytorch.png")
plt.close() 

plt.figure(figsize=(12, 6))
plt.hist(rollout_rewards, bins=30, edgecolor='black')
plt.title(f'{ENV_NAME} - Histogram of 500 Rollout Episodes (PyTorch)')
plt.xlabel('Episode Reward')
plt.ylabel('Frequency')
plt.axvline(mean_reward, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_reward:.2f}')
plt.legend()
plt.grid(True)
plt.savefig(f"{ENV_NAME}_histogram_pytorch.png")
plt.close()