import numpy as np

class Evaluator:

    def __init__(self):
        self.total_reward = 0
        self.total_polls = 0
        self.total_delay = 0
        self.violation_count = 0
        self.steps = 0

    def update(self, reward, interval, apdu_flag, delay):

        self.total_reward += reward
        self.total_polls += 1
        self.total_delay += delay

        if delay > 30:
            self.violation_count += 1

        self.steps += 1

    def results(self):

        avg_delay = self.total_delay / max(1, self.steps)

        violation_rate = self.violation_count / max(1, self.steps)

        return {
            "total_reward": self.total_reward,
            "total_polls": self.total_polls,
            "avg_delay": avg_delay,
            "violation_rate": violation_rate
        }