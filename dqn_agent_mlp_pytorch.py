import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_shape, num_actions, discount_factor, replay_buffer_size=10000, lr=1e-3):
        self.state_dim = state_shape[0]
        self.num_actions = num_actions
        self.gamma = discount_factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 
        self.lr = lr

        self.main_network = QNetwork(self.state_dim, self.num_actions).to(device)
        self.target_network = QNetwork(self.state_dim, self.num_actions).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval() 

        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.criterion = nn.MSELoss() 

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad(): 
            q_values = self.main_network(state_tensor)
        
        return torch.argmax(q_values, dim=1).item()

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None 

        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(device)

        q_values = self.main_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item() 

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay