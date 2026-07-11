import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from env.polling_env import PollingEnv
from rl.ddqn_agent import DDQNAgent

EPISODES = 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=("dqn", "ddqn"),
        default="ddqn",
    )
    args = parser.parse_args()

    results_dir = Path("results")
    models_dir = results_dir / "models"
    plots_dir = results_dir / "plots"

    for directory in (results_dir, models_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = PollingEnv()

    state_dim = len(env.reset())
    action_dim = 5

    agent = DDQNAgent(state_dim, action_dim, algorithm=args.algorithm)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    reward_history = []

    for ep in range(EPISODES):

        state = env.reset()
        done = False
        total_reward = 0

        while not done:

            action = agent.select_action(state, epsilon)

            next_state, reward, done = env.step(action)

            agent.replay_buffer.push(
                (state, action, reward, next_state, done)
            )

            agent.train_step()

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        reward_history.append(total_reward)

        if ep % 50 == 0:
            print(
                f"Episode {ep}, Reward {total_reward}, Epsilon {epsilon:.4f}"
            )

    plt.plot(reward_history)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(plots_dir / f"{args.algorithm}_reward_curve.png")
    plt.show()

    np.save(
        results_dir / f"{args.algorithm}_reward_history.npy",
        np.asarray(reward_history),
    )
    torch.save(
        agent.q_net.state_dict(),
        models_dir / f"{args.algorithm}_trained_model.pth",
    )


if __name__ == "__main__":
    main()
