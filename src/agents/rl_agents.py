"""Deep RL agents for gate library subset selection.

GateSelectionEnv wraps TechMapper as a Gymnasium environment where each step
selects one gate until sample_gate gates have been chosen, then evaluates.

DQNAgent  — standard DQN with experience replay.
DDQNAgent — Double DQN with soft target-network updates.
Both share the same _QNetwork architecture and update_batch() interface.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class GateSelectionEnv(gym.Env):
    """Gymnasium env: select sample_gate gates one-by-one, evaluate on completion.

    Observation: binary vector of length total_gates (1 = selected).
    Action:      integer gate index.
    Reward:      0 while selecting; TechMapper.calculate_reward() when done.
    step() returns (next_state, reward, done, delay, area).
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, mapper, sample_gate):
        super().__init__()
        self.mapper = mapper
        self.total_gates = mapper.num_arms
        self.sample_gate = sample_gate
        self.action_space = spaces.Discrete(self.total_gates)
        self.observation_space = spaces.MultiBinary(self.total_gates)
        self.state = np.zeros(self.total_gates, dtype=int)
        self.selection_count = 0

    def step(self, action):
        if self.state[action] == 0 and self.selection_count < self.sample_gate:
            self.state[action] = 1
            self.selection_count += 1

        done = self.selection_count == self.sample_gate
        reward, delay, area = 0, 1, 1
        next_state = self.state.copy()

        if done:
            selected = list(np.where(self.state == 1)[0])
            delay, area = self.mapper.map_subset(selected, tag="rl")
            reward = self.mapper.calculate_reward(delay, area)
            next_state = self.reset()

        return next_state, reward, done, delay, area

    def reset(self):
        self.state = np.zeros(self.total_gates, dtype=int)
        self.selection_count = 0
        return self.state

    def render(self, mode="human"):
        print(f"Selected Gates: {np.where(self.state == 1)[0]}")

    def close(self):
        pass


class _QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Standard DQN with a single Q-network and experience-replay batches."""

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma

    def select_action(self, state, epsilon=0.2):
        if np.random.rand() < epsilon:
            return np.random.randint(0, len(state))
        with torch.no_grad():
            q = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            return q.argmax().item()

    def update_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_qs = self.model(states).gather(1, actions).squeeze(1)
        next_qs = self.model(next_states).max(1)[0]
        expected_qs = rewards + self.gamma * (1 - dones) * next_qs

        loss = F.mse_loss(current_qs, expected_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DDQNAgent:
    """Double DQN: online network selects actions, target network evaluates them.

    Target network is soft-updated each batch step via tau.
    """

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, tau=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = _QNetwork(state_size, action_size).to(self.device)
        self.target = _QNetwork(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state, epsilon=0.2):
        if np.random.rand() < epsilon:
            return np.random.randint(0, len(state))
        with torch.no_grad():
            q = self.online(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            return q.argmax().item()

    def update_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_qs = self.online(states).gather(1, actions).squeeze(1)
        next_actions = self.online(next_states).argmax(1).unsqueeze(1)
        next_qs = self.target(next_states).gather(1, next_actions).squeeze(1)
        expected_qs = rewards + self.gamma * (1 - dones) * next_qs

        loss = F.mse_loss(current_qs, expected_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)
