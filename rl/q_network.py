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