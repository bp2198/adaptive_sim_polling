import torch
import torch.optim as optim
import numpy as np

from rl.q_network import QNetwork
from rl.replay_buffer import PrioritizedReplayBuffer


class DDQNAgent:

    def __init__(self, state_dim, action_dim, algorithm="ddqn"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm.lower()

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
            return np.random.randint(0, self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state)

        return torch.argmax(q_values).item()

    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch, indices, weights = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # -----------------------------
        # DQN Target
        # -----------------------------
        if self.algorithm == "dqn":

            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]

        # -----------------------------
        # DDQN Target
        # -----------------------------
        else:

            with torch.no_grad():
                next_actions = torch.argmax(
                    self.q_net(next_states),
                    dim=1
                )

                next_q = self.target_net(next_states).gather(
                    1,
                    next_actions.unsqueeze(1)
                ).squeeze(1)

        target_q = rewards + self.gamma * next_q * (1 - dones)

        td_errors = target_q - current_q
        importance_weights = torch.FloatTensor(weights).to(self.device)
        loss = (importance_weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(
            indices,
            td_errors.detach().cpu().numpy()
        )

        self.step_count += 1

        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(
                self.q_net.state_dict()
            )

        return loss.item()
