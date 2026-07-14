import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    metrics_dir = Path("results") / "metrics"
    comparisons_dir = Path("results") / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for metrics_path in sorted(metrics_dir.glob("*.json")):
        with metrics_path.open("r") as file:
            metrics = json.load(file)

        records.append(
            {
                "Algorithm": metrics["algorithm"],
                "Network": metrics["network"],
                "Seed": metrics["seed"],
                "Episodes": metrics["episodes"],
                "TrainingTime": metrics["training_time"],
                "FinalReward": metrics["final_reward"],
                "AverageRewardLast100": metrics[
                    "average_reward_last100"
                ],
                "MaxReward": metrics["max_reward"],
                "MeanReward": metrics["mean_reward"],
            }
        )

    columns = [
        "Algorithm",
        "Network",
        "Seed",
        "Episodes",
        "TrainingTime",
        "FinalReward",
        "AverageRewardLast100",
        "MaxReward",
        "MeanReward",
    ]
    dataframe = pd.DataFrame(records, columns=columns)

    print(dataframe)

    aggregated_dataframe = (
        dataframe.groupby(["Algorithm", "Network"], as_index=False)
        .agg(
            NumberOfRuns=("Seed", "count"),
            MeanTrainingTime=("TrainingTime", "mean"),
            StdTrainingTime=("TrainingTime", "std"),
            MeanFinalReward=("FinalReward", "mean"),
            StdFinalReward=("FinalReward", "std"),
            MeanAverageRewardLast100=("AverageRewardLast100", "mean"),
            StdAverageRewardLast100=("AverageRewardLast100", "std"),
            BestAverageRewardLast100=("AverageRewardLast100", "max"),
            WorstAverageRewardLast100=("AverageRewardLast100", "min"),
            MeanReward=("MeanReward", "mean"),
        )
        .sort_values("MeanAverageRewardLast100", ascending=False)
        .reset_index(drop=True)
    )
    floating_columns = aggregated_dataframe.select_dtypes(
        include="float"
    ).columns
    aggregated_dataframe[floating_columns] = aggregated_dataframe[
        floating_columns
    ].round(3)

    print(aggregated_dataframe)

    dataframe.to_csv(comparisons_dir / "summary.csv", index=False)
    with (comparisons_dir / "summary.md").open("w") as file:
        file.write("| " + " | ".join(columns) + " |\n")
        file.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in dataframe.itertuples(index=False, name=None):
            values = [str(value).replace("|", "\\|") for value in row]
            file.write("| " + " | ".join(values) + " |\n")
        file.write("\n")

    aggregated_summary_path = comparisons_dir / "aggregated_summary.csv"
    aggregated_dataframe.to_csv(aggregated_summary_path, index=False)
    with (comparisons_dir / "aggregated_summary.md").open("w") as file:
        file.write(
            "| "
            + " | ".join(aggregated_dataframe.columns)
            + " |\n"
        )
        file.write(
            "|" + "|".join("---" for _ in aggregated_dataframe.columns) + "|\n"
        )
        for row in aggregated_dataframe.itertuples(index=False, name=None):
            values = [str(value).replace("|", "\\|") for value in row]
            file.write("| " + " | ".join(values) + " |\n")
        file.write("\n")

    if not aggregated_dataframe.empty:
        best_configuration = aggregated_dataframe.iloc[0]
        print("\n" + "=" * 34)
        print("Best Performing Configuration")
        print(f"Algorithm: {best_configuration['Algorithm']}")
        print(f"Network: {best_configuration['Network']}")
        print(
            "Mean Avg Reward: "
            f"{best_configuration['MeanAverageRewardLast100']:.3f}"
        )
        print(
            "Std Dev: "
            f"{best_configuration['StdAverageRewardLast100']:.3f}"
        )
        print("=" * 34)

    plt.figure(figsize=(10, 6))
    for reward_path in sorted(Path("results").glob("*_reward.npy")):
        rewards = np.load(reward_path)
        experiment_name = reward_path.stem
        episodes = np.arange(1, len(rewards) + 1)

        plt.plot(
            episodes,
            rewards,
            alpha=0.3,
            label="_nolegend_",
        )

        if len(rewards) >= 50:
            moving_average = np.convolve(
                rewards,
                np.ones(50) / 50,
                mode="valid",
            )
            plt.plot(
                np.arange(50, len(rewards) + 1),
                moving_average,
                linewidth=2,
                label=experiment_name,
            )

    plt.title("Reward Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparisons_dir / "reward_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
