import numpy as np
import matplotlib.pyplot as plt

# load real RSRP
real = []
with open("analysis/real_rsrp_trace.txt") as f:
    for line in f:
        real.append(float(line.strip()))

# synthetic signal model
synthetic = np.random.normal(loc=-92, scale=10, size=1000)

plt.figure(figsize=(7,5))

# synthetic histogram
plt.hist(synthetic,
         bins=30,
         alpha=0.6,
         color="orange",
         label="Synthetic simulator")

# real histogram (outline)
plt.hist(real,
         bins=5,
         histtype="step",
         linewidth=2,
         color="blue",
         label="Real modem logs")

plt.xlabel("RSRP (dBm)")
plt.ylabel("Frequency")
plt.title("Comparison of Real and Synthetic Signal Distributions")

plt.legend()

plt.savefig("real_vs_synthetic_rsrp.png", dpi=300)

plt.show()