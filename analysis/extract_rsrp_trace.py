import re
input_file = "analysis/oos_rsrp_logs.txt"
output_file = "analysis/real_rsrp_trace.txt"

rsrp_values = []

with open(input_file, "r") as f:
    for line in f:
        match = re.search(r'Cell Quality RSRP = (-?\d+\.\d+)', line)
        if match:
            rsrp_values.append(float(match.group(1)))

print("Total RSRP samples extracted:", len(rsrp_values))

with open(output_file, "w") as f:
    for v in rsrp_values:
        f.write(str(v) + "\n")

print("Saved to:", output_file)