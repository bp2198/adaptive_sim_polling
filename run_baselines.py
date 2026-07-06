import numpy as np
from env.polling_env import PollingEnv
from baselines.static import StaticBaseline
from baselines.heuristic import HeuristicBaseline
from metrics.evaluator import Evaluator
from config import EPISODE_LENGTH, POLL_INTERVALS

def run_policy(policy, episodes=50):

    env = PollingEnv()
    evaluator = Evaluator()

    for ep in range(episodes):

        state = env.reset()
        done = False

        while not done:

            action = policy.act(state)

            interval = POLL_INTERVALS[action]

            next_state, reward, done = env.step(action)

            apdu_flag = state[5]

            delay = env.apdu_wait

            evaluator.update(reward, interval, apdu_flag, delay)

            state = next_state

    return evaluator.results()


if __name__ == "__main__":

    static_policy = StaticBaseline(interval=10)
    heuristic_policy = HeuristicBaseline()

    static_results = run_policy(static_policy)
    heuristic_results = run_policy(heuristic_policy)

    print("Static Baseline:", static_results)
    print("Heuristic Baseline:", heuristic_results)