#!/usr/bin/env python
"""
crossq_agent.py

A CrossQ-style agent implementation that interacts with the SimRacingEnv.
This implementation discretizes the 3-dimensional continuous action space into 27 discrete actions.
The Q-network is widened and uses Batch Normalization.
No target network is used – instead, we perform a combined forward pass
through the network for both current and next states.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import itertools
import simracing_env  # Import the Gym environment defined earlier


def get_discrete_actions():
    """
    Discretize each control into three levels: -1.0, 0.0, and 1.0.
    Return an array of all possible combinations (27 total actions).
    """
    discrete_vals = [-1.0, 0.0, 1.0]
    actions = list(itertools.product(discrete_vals, repeat=3))
    return np.array(actions, dtype=np.float32)  # shape (27, 3)


class CrossQNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(CrossQNetwork, self).__init__()
        # Widened network with BatchNorm
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        # x is of shape (batch_size, input_dim)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


def train_crossq(env, model, num_episodes=500, gamma=0.99, epsilon=0.1, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer()
    discrete_actions = get_discrete_actions()
    num_actions = len(discrete_actions)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # Epsilon-greedy action selection:
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(num_actions)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                action_idx = int(torch.argmax(q_values, dim=1).item())
            action = discrete_actions[action_idx]

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action_idx, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step_count += 1

            if len(replay_buffer) >= batch_size:
                # Sample a batch of transitions.
                batch = replay_buffer.sample(batch_size)
                states, action_indices, rewards, next_states, dones = zip(*batch)
                states_tensor = torch.FloatTensor(states)
                rewards_tensor = torch.FloatTensor(rewards)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones)
                action_indices_tensor = torch.LongTensor(action_indices).unsqueeze(1)

                # --- CrossQ update: Combined forward pass for BN ---
                # Concatenate states and next_states along the batch dimension.
                combined_input = torch.cat([states_tensor, next_states_tensor], dim=0)
                combined_q = model(combined_input)
                # Split back into q-values for current states and next states.
                q_current = combined_q[:batch_size]
                q_next = combined_q[batch_size:]
                # For the current states, select the Q-values corresponding to taken actions.
                q_current_taken = q_current.gather(1, action_indices_tensor).squeeze(1)
                # For next states, use the maximum Q-value over actions.
                q_next_max, _ = torch.max(q_next, dim=1)
                targets = rewards_tensor + gamma * q_next_max * (1 - dones_tensor)
                
                loss = criterion(q_current_taken, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f} | Steps: {step_count}")
    

if __name__ == "__main__":
    # Instantiate the environment.
    env = simracing_env.SimRacingEnv()
    input_dim = env.observation_space.shape[0]  # e.g., 5 features
    discrete_actions = get_discrete_actions()
    num_actions = len(discrete_actions)  # e.g., 27 discrete actions

    # Initialize the CrossQ model.
    model = CrossQNetwork(input_dim, num_actions)

    # Train the CrossQ agent.
    train_crossq(env, model, num_episodes=200)

    # Optionally, save the model after training.
    torch.save(model.state_dict(), "crossq_model.pth")
