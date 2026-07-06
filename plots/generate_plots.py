import matplotlib.pyplot as plt

# Results from experiments
methods = ["Static", "Heuristic", "RL"]

rewards = [12208, 4639, 11166]
delay = [0.29, 1.01, 0.43]
violations = [0.0, 0.0168, 0.0029]

# -------- Reward comparison --------
plt.figure()

plt.bar(methods, rewards)

plt.title("Reward Comparison Across Policies")
plt.ylabel("Total Reward")

plt.savefig("plots/reward_comparison.png", dpi=300)

plt.show()


# -------- Delay comparison --------
plt.figure()

plt.bar(methods, delay)

plt.title("Average Proactive Command Delay")
plt.ylabel("Delay")

plt.savefig("plots/delay_comparison.png", dpi=300)

plt.show()

# -------- Reward comparison -------- 

plt.bar(methods, violations)

plt.title("Constraint Violation Rate")
plt.ylabel("Violation Rate")

plt.savefig("plots/violation_comparison.png", dpi=300)

plt.show()