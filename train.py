import argparse
import csv
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
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    results_dir = Path("results")
    models_dir = results_dir / "models"
    plots_dir = results_dir / "plots"
    metrics_dir = results_dir / "metrics"
    csv_dir = results_dir / "csv"

    for directory in (results_dir, models_dir, plots_dir, metrics_dir, csv_dir):
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
    training_start = time.time()

    for ep in range(args.episodes):

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

    training_time_seconds = time.time() - training_start

    rewards = np.asarray(reward_history, dtype=float)
    moving_average_50 = np.full(args.episodes, np.nan)
    if args.episodes >= 50:
        moving_average_50[49:] = np.convolve(
            rewards, np.ones(50) / 50, mode="valid"
        )

    episodes = np.arange(1, args.episodes + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label="Raw Reward")
    plt.plot(episodes, moving_average_50, label="50 Episode Moving Average")
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        plots_dir / f"{experiment_name}_reward_curve.png",
        dpi=300,
    )
    plt.close()

    np.save(
        results_dir / f"{experiment_name}_reward.npy",
        rewards,
    )
    model_path = models_dir / f"{experiment_name}_model.pth"
    torch.save(
        agent.q_net.state_dict(),
        model_path,
    )

    with (csv_dir / f"{experiment_name}_reward.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward", "MovingAverage50"])
        writer.writerows(zip(episodes, rewards, moving_average_50))

    metrics = {
        "algorithm": args.algorithm,
        "network": args.network,
        "episodes": args.episodes,
        "seed": args.seed,
        "training_time": training_time_seconds,
        "final_reward": float(rewards[-1]),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "average_reward_last100": float(np.mean(rewards[-100:])),
    }
    with (metrics_dir / f"{experiment_name}_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    print("\n" + "=" * 36)
    print(f"Algorithm : {args.algorithm.upper()}")
    print(f"Network : {args.network.title()}")
    print(f"Episodes : {args.episodes}")
    print(f"Seed : {args.seed}")
    print(f"Training Time : {training_time_seconds:.2f} seconds")
    print(f"Final Reward : {metrics['final_reward']:.4f}")
    print(
        "Average Reward Last100 : "
        f"{metrics['average_reward_last100']:.4f}"
    )
    print(f"Model Saved : {model_path}")
    print("=" * 36)


if __name__ == "__main__":
    main()
