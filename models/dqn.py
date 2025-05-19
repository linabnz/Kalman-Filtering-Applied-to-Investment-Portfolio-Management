# models/dqn.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepQNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        
        return self.layers(x)


class TradingStrategyOptimizer:
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        
        self.q_network = DeepQNetwork(state_dim, action_dim)
        self.target_network = DeepQNetwork(state_dim, action_dim)
        self.update_target_network()
        
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        
        self.memory = []
        self.batch_size = 64
        self.max_memory_size = 10000
    
    def update_target_network(self):
        
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
    
    def act(self, state, epsilon=0.0):
        
        if np.random.random() < epsilon:
            
            return np.random.choice(self.action_dim)
        else:
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).detach().numpy()[0]
            return np.argmax(q_values)
    
    def train(self):
        
        if len(self.memory) < self.batch_size:
            return
        
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in batch:
            s, a, r, n_s, d = self.memory[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(n_s)
            dones.append(d)
        
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
       
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()