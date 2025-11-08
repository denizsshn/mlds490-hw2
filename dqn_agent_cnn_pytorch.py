import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        flattened_size = 64 * 7 * 6
        
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.reshape(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)

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

class AtariDQNAgent:
    def __init__(self, state_shape, num_actions, discount_factor, lr, replay_buffer_size):
        self.state_shape = state_shape 
        self.num_actions = num_actions
        self.gamma = discount_factor
        self.lr = lr

        self.main_network = QNetwork(self.state_shape, self.num_actions).to(device)
        self.target_network = QNetwork(self.state_shape, self.num_actions).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval() 

        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.criterion = nn.MSELoss() 

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
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

        torch.nn.utils.clip_grad_value_(self.main_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item() 

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())