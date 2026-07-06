from env.polling_env import PollingEnv
import numpy as np

env = PollingEnv()

state = env.reset()

for _ in range(20):

    action = np.random.randint(0,5)

    state, reward, done = env.step(action)

    print(state, reward)

    if done:
        break