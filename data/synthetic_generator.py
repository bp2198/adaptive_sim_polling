import numpy as np
from config import *

class SyntheticSignalGenerator:

    def __init__(self):
        self.rsrp = RSRP_MEAN
        self.oos_counter = 0

    def step_signal(self):
        noise = np.random.normal(0, RSRP_STD)
        self.rsrp += noise * 0.2

        # clamp
        self.rsrp = np.clip(self.rsrp, -120, -70)

        return self.rsrp

    def update_oos(self):
        if self.oos_counter > 0:
            self.oos_counter -= 1
            return 1

        if np.random.rand() < OOS_PROB:
            self.oos_counter = int(np.random.exponential(OOS_DURATION_MEAN))
            return 1

        return 0

    def rat_from_signal(self, rsrp):
        if rsrp > -85:
            return 2  # NR
        elif rsrp > -100:
            return 1  # LTE
        else:
            return 0  # GSM

    def apdu_event(self, oos_flag):
        prob = APDU_BASE_PROB

        if oos_flag:
            prob *= 2

        return int(np.random.rand() < prob)

    def step(self):

        rsrp = self.step_signal()
        oos = self.update_oos()
        rat = self.rat_from_signal(rsrp)
        apdu = self.apdu_event(oos)

        return rsrp, rat, oos, apdu