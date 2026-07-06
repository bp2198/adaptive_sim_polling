import numpy as np
from config import POLL_INTERVALS

class StaticBaseline:

    def __init__(self, interval=10):
        self.interval = interval
        self.action = POLL_INTERVALS.index(interval)

    def act(self, state):
        return self.action