import numpy as np
from config import POLL_INTERVALS

class HeuristicBaseline:

    def act(self, state):

        rsrp = state[0]
        oos = state[4]
        apdu = state[5]

        if apdu:
            return 0  # 5s polling

        if oos:
            return 4  # 60s polling

        if rsrp > 0.7:
            return 3  # 40s polling

        return 1  # 10s polling