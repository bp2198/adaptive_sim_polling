import numpy as np
import matplotlib.pyplot as plt
import torch
from env.polling_env import PollingEnv
from rl.ddqn_agent import DDQNAgent

EPISODES = 1000

env = PollingEnv()

state_dim = len(env.reset())
action_dim = 5

agent = DDQNAgent(state_dim, action_dim)

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
        print(f"Episode {ep}, Reward {total_reward}")

plt.plot(reward_history)
plt.title("Training Reward Curve")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

torch.save(agent.q_net.state_dict(), "trained_model.pth")