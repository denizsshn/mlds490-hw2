import torch

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from collections import deque
from dqn_agent_cnn_pytorch import AtariDQNAgent, device 
import ale_py, shimmy
import time

gym.register_envs(ale_py)


def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.mean(axis=2)
    mspacman_color_mean = 148.45 
    img[img == mspacman_color_mean] = 0
    img = (img / 255.0).astype(np.float32)
    return img.reshape(88, 80, 1)

def stack_frames(stacked_deque, new_frame, is_new_episode):
    if is_new_episode:
        stacked_deque.clear()
        for _ in range(4):
            stacked_deque.append(new_frame)
    else:
        stacked_deque.append(new_frame)
    stacked_state = np.concatenate(stacked_deque, axis=2) 
    return stacked_state, stacked_deque

ENV_NAME = "MsPacman-v0"
DISCOUNT_FACTOR = 0.99
NUM_EPISODES = 2000     
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 1000 
LEARNING_RATE = 0.00025   
MIN_REPLAY_SIZE = 5000    
REPLAY_BUFFER_SIZE = 100_000 

EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 250_000 

env = gym.make(ENV_NAME)
STATE_SHAPE = (88, 80, 4) 
NUM_ACTIONS = env.action_space.n 

agent = AtariDQNAgent(STATE_SHAPE, NUM_ACTIONS, DISCOUNT_FACTOR, LEARNING_RATE, REPLAY_BUFFER_SIZE)
frame_deque = deque(maxlen=4)
total_steps = 0

all_episode_rewards = []
all_max_q_values = []
rewards_window = deque(maxlen=100) 
all_moving_avgs = []

start_time = time.time()

for episode in range(NUM_EPISODES):
    frame, info = env.reset()
    frame = preprocess_observation(frame)
    state, frame_deque = stack_frames(frame_deque, frame, is_new_episode=True)
    
    episode_reward = 0
    max_q_this_episode = -float('inf')
    done = False
    
    while not done:
        total_steps += 1
        
        epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (total_steps / EPSILON_DECAY_STEPS))
        
        action = agent.choose_action(state, epsilon)
        
        next_frame, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_frame = preprocess_observation(next_frame)
        next_state, frame_deque = stack_frames(frame_deque, next_frame, is_new_episode=False)
        
        agent.replay_buffer.add(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward

        if len(agent.replay_buffer) > MIN_REPLAY_SIZE:
            loss = agent.train(BATCH_SIZE)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values_pred = agent.main_network(state_tensor).cpu().numpy()[0]
                max_q_this_episode = max(max_q_this_episode, np.max(q_values_pred))
        
        if total_steps % TARGET_UPDATE_FREQ == 0 and len(agent.replay_buffer) > MIN_REPLAY_SIZE:
            agent.update_target_network()
            
    all_episode_rewards.append(episode_reward)
    rewards_window.append(episode_reward)
    all_moving_avgs.append(np.mean(rewards_window))
    if max_q_this_episode > -float('inf'):
        all_max_q_values.append(max_q_this_episode)


end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

rollout_rewards = []
for i in range(500):
    frame, info = env.reset()
    frame = preprocess_observation(frame)
    state, frame_deque = stack_frames(frame_deque, frame, is_new_episode=True)
    
    episode_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = agent.main_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        
        next_frame, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_frame = preprocess_observation(next_frame)
        state, frame_deque = stack_frames(frame_deque, next_frame, is_new_episode=False)
        
        episode_reward += reward
            
    rollout_rewards.append(episode_reward)

mean_reward = np.mean(rollout_rewards)
std_reward = np.std(rollout_rewards)
print(f"Rollout Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(all_max_q_values)
plt.title(f'{ENV_NAME} - Max Q-Value vs. Training Episodes (PyTorch)')
plt.xlabel('Training Episode (after training started)')
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