import argparse
import json
from pathlib import Path
import random
import time

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
    parser.add_argument(
        "--network",
        choices=("standard", "dueling"),
        default="standard",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results_dir = Path("results")
    models_dir = results_dir / "models"
    plots_dir = results_dir / "plots"
    metrics_dir = results_dir / "metrics"

    for directory in (results_dir, models_dir, plots_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    experiment_name = f"{args.algorithm}_{args.network}_seed{args.seed}"

    env = PollingEnv()

    state_dim = len(env.reset())
    action_dim = 5

    agent = DDQNAgent(
        state_dim,
        action_dim,
        algorithm=args.algorithm,
        network_type=args.network,
    )

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    reward_history = []
    training_start = time.perf_counter()

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

    training_time = time.perf_counter() - training_start

    plt.plot(reward_history)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(plots_dir / f"{experiment_name}_reward.png")
    plt.close()

    np.save(
        results_dir / f"{experiment_name}_reward.npy",
        np.asarray(reward_history),
    )
    torch.save(
        agent.q_net.state_dict(),
        models_dir / f"{experiment_name}_model.pth",
    )

    metrics = {
        "algorithm": args.algorithm,
        "network": args.network,
        "episodes": EPISODES,
        "seed": args.seed,
        "training_time": training_time,
        "final_reward": float(reward_history[-1]),
        "average_reward_last100": float(np.mean(reward_history[-100:])),
    }
    with (metrics_dir / f"{experiment_name}_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()
