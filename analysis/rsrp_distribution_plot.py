import matplotlib.pyplot as plt

rsrp = []

with open("analysis/real_rsrp_trace.txt") as f:
    for line in f:
        rsrp.append(float(line.strip()))

plt.hist(rsrp, bins=10)

plt.title("RSRP Distribution from Real Modem Logs")
plt.xlabel("RSRP (dBm)")
plt.ylabel("Frequency")

plt.savefig("real_rsrp_distribution.png", dpi=300)

plt.show()