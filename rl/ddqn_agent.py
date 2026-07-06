import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from rl.q_network import QNetwork
from rl.replay_buffer import PrioritizedReplayBuffer


class DDQNAgent:

    def __init__(self, state_dim, action_dim):

        self.device = torch.device("cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

        self.replay_buffer = PrioritizedReplayBuffer()

        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 200

        self.step_count = 0

    def select_action(self, state, epsilon):

        if np.random.rand() < epsilon:
            return np.random.randint(0, 5)

        state = torch.FloatTensor(state).unsqueeze(0)

        q_values = self.q_net(state)

        return torch.argmax(q_values).item()

    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)

        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        next_actions = torch.argmax(self.q_net(next_states), dim=1)

        next_q = self.target_net(next_states).gather(
            1, next_actions.unsqueeze(1)).squeeze()

        target = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1

        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())