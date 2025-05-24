import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.qnetwork import QNetwork

class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, gamma=0.99):
        self.q_net = QNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.action_space = list(range(output_dim))

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space)
        state_tensor = torch.tensor(state).unsqueeze(0).float()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state).unsqueeze(0).float()
        next_state_tensor = torch.tensor(next_state).unsqueeze(0).float()

        with torch.no_grad():
            q_next = self.q_net(next_state_tensor)
            q_target = reward + self.gamma * torch.max(q_next).item() if not done else reward

        q_pred = self.q_net(state_tensor)[0, action]
        loss = self.loss_fn(q_pred, torch.tensor(q_target).float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.q_net.eval()
