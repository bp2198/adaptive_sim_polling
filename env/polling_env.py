import numpy as np
from config import *
from data.synthetic_generator import SyntheticSignalGenerator

class PollingEnv:

    def __init__(self):

        self.generator = SyntheticSignalGenerator()
        self.step_count = 0
        self.prev_interval = 10
        self.apdu_wait = 0

    def reset(self):

        self.generator = SyntheticSignalGenerator()
        self.step_count = 0
        self.prev_interval = 10
        self.apdu_wait = 0

        rsrp, rat, oos, apdu = self.generator.step()

        state = self.build_state(rsrp, rat, oos, apdu)

        return state

    def build_state(self, rsrp, rat, oos, apdu):

        rat_onehot = np.zeros(3)
        rat_onehot[rat] = 1

        state = np.array([
            (rsrp + 120) / 50,
            *rat_onehot,
            oos,
            apdu,
            self.prev_interval / MAX_INTERVAL
        ])

        return state

    def compute_reward(self, interval, apdu):

        energy = (MAX_INTERVAL - interval) / MAX_INTERVAL

        delay_penalty = 0
        violation_penalty = 0

        if apdu:
            self.apdu_wait += interval

            delay_penalty = self.apdu_wait / MAX_DELAY

            if self.apdu_wait > MAX_DELAY:
                violation_penalty = 1

        else:
            self.apdu_wait = 0

        reward = (
            W_ENERGY * energy
            - W_DELAY * delay_penalty
            - W_VIOLATION * violation_penalty
        )

        return reward

    def step(self, action):

        interval = POLL_INTERVALS[action]

        rsrp, rat, oos, apdu = self.generator.step()

        reward = self.compute_reward(interval, apdu)

        self.prev_interval = interval
        self.step_count += 1

        done = self.step_count >= EPISODE_LENGTH

        state = self.build_state(rsrp, rat, oos, apdu)

        return state, reward, done