import numpy as np

# Action space (polling intervals in seconds)
POLL_INTERVALS = [5, 10, 20, 40, 60]

# Episode settings
EPISODE_LENGTH = 300  # steps per episode
NUM_EPISODES = 1000

# Signal parameters
RSRP_MEAN = -95
RSRP_STD = 6

# OOS parameters
OOS_PROB = 0.05
OOS_DURATION_MEAN = 10

# APDU event probability
APDU_BASE_PROB = 0.02

# Reward weights
W_ENERGY = 1.0
W_DELAY = 2.0
W_VIOLATION = 5.0

MAX_DELAY = 30
MAX_INTERVAL = max(POLL_INTERVALS)