import numpy as np
import random

class PrioritizedReplayBuffer:

    def __init__(self, capacity=50000, alpha=0.6):

        self.capacity = capacity
        self.alpha = alpha

        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, transition, td_error=1.0):

        priority = (abs(td_error) + 1e-5) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        probs = np.array(self.priorities) / sum(self.priorities)

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        samples = [self.buffer[i] for i in indices]

        return samples

    def __len__(self):
        return len(self.buffer)