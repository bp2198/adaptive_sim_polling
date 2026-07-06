from env.polling_env import PollingEnv
from rl.ddqn_agent import DDQNAgent
from metrics.evaluator import Evaluator
from config import POLL_INTERVALS

import torch

def evaluate(agent, episodes=50):

    env = PollingEnv()
    evaluator = Evaluator()

    for ep in range(episodes):

        state = env.reset()
        done = False

        while not done:

            action = agent.select_action(state, epsilon=0)

            interval = POLL_INTERVALS[action]

            next_state, reward, done = env.step(action)

            apdu_flag = state[5]
            delay = env.apdu_wait

            evaluator.update(reward, interval, apdu_flag, delay)

            state = next_state

    return evaluator.results()


if __name__ == "__main__":

    env = PollingEnv()
    state_dim = len(env.reset())
    action_dim = 5

    agent = DDQNAgent(state_dim, action_dim)

    agent.q_net.load_state_dict(torch.load("trained_model.pth"))

    results = evaluate(agent)

    print("RL Agent:", results)