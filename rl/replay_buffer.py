import numpy as np


class PrioritizedReplayBuffer:

    def __init__(
        self,
        capacity=50000,
        alpha=0.6,
        beta=0.4,
        beta_increment=1e-3,
        priority_epsilon=1e-5,
    ):

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priority_epsilon = priority_epsilon

        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, transition, td_error=None):

        if td_error is None:
            priority = max(self.priorities, default=1.0)
        else:
            priority = self._priority_from_error(td_error)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):

        priorities = np.asarray(self.priorities, dtype=np.float64)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(
            len(self.buffer), batch_size, p=probabilities
        )

        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):

        for index, td_error in zip(indices, td_errors):
            self.priorities[int(index)] = self._priority_from_error(td_error)

    def _priority_from_error(self, td_error):

        return (abs(float(td_error)) + self.priority_epsilon) ** self.alpha

    def __len__(self):
        return len(self.buffer)
