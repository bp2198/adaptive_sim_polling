import random

import numpy as np
import torch

from env.polling_env import PollingEnv
from rl.q_network import DuelingQNetwork


STATE_DIM = 7
ACTION_DIM = 5
EPISODES = 100

FP32_MODEL = "results/models/ddqn_dueling_seed42_model.pth"
INT8_MODEL = "results/models/ddqn_dueling_seed42_quantized.pth"


def evaluate(model, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = PollingEnv()

    rewards = []

    model.eval()

    with torch.no_grad():

        for _ in range(EPISODES):

            state = env.reset()

            done = False

            total_reward = 0

            while not done:

                state_tensor = (
                    torch.FloatTensor(state)
                    .unsqueeze(0)
                )

                q_values = model(state_tensor)

                action = torch.argmax(q_values).item()

                next_state, reward, done = env.step(action)

                total_reward += reward

                state = next_state

            rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)


# -----------------------------
# FP32
# -----------------------------

fp32_model = DuelingQNetwork(
    STATE_DIM,
    ACTION_DIM,
)

fp32_model.load_state_dict(
    torch.load(
        FP32_MODEL,
        map_location="cpu",
    )
)

fp32_mean, fp32_std = evaluate(fp32_model, seed=42)

# -----------------------------
# INT8
# -----------------------------

int8_model = DuelingQNetwork(
    STATE_DIM,
    ACTION_DIM,
)

int8_model = torch.quantization.quantize_dynamic(
    int8_model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

int8_model.load_state_dict(
    torch.load(
        INT8_MODEL,
        map_location="cpu",
    )
)

int8_mean, int8_std = evaluate(int8_model, seed=42)

print("\n===============================")
print("Policy Evaluation")
print("===============================")

print(
    f"FP32 Average Reward : "
    f"{fp32_mean:.2f} ± {fp32_std:.2f}"
)

print(
    f"INT8 Average Reward : "
    f"{int8_mean:.2f} ± {int8_std:.2f}"
)

print(
    f"Difference : "
    f"{abs(fp32_mean-int8_mean):.2f}"
)

percentage_difference = (
    abs(fp32_mean - int8_mean) / fp32_mean
) * 100

print(
    f"Percentage Difference : "
    f"{percentage_difference:.2f}%"
)

print("===============================")
