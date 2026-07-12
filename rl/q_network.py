import torch
import torch.nn as nn

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.model(x)


class DuelingQNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()

        # Shared feature extractor used by both value and advantage streams.
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Value stream estimates V(s).
        self.value_stream = nn.Linear(64, 1)

        # Advantage stream estimates A(s, a) for each action.
        self.advantage_stream = nn.Linear(64, action_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Aggregate Q(s, a) = V(s) + A(s, a) - mean(A(s, a)).
        return value + advantage - advantage.mean(dim=1, keepdim=True)
