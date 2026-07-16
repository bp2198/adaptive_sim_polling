import time
from pathlib import Path

import torch

from rl.q_network import DuelingQNetwork

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "results/models/ddqn_dueling_seed42_model.pth"

STATE_DIM = 7
ACTION_DIM = 5

# --------------------------------------------------
# Load Original Model
# --------------------------------------------------

model = DuelingQNetwork(
    STATE_DIM,
    ACTION_DIM,
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
    )
)

model.eval()

# --------------------------------------------------
# Original Model Size
# --------------------------------------------------

original_size = Path(MODEL_PATH).stat().st_size / 1024

# --------------------------------------------------
# Dynamic Quantization
# --------------------------------------------------

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

# --------------------------------------------------
# Output Comparison
# --------------------------------------------------

dummy_input = torch.randn(1, STATE_DIM)

with torch.no_grad():
    fp32_output = model(dummy_input)
    int8_output = quantized_model(dummy_input)

print("\nFP32 Output:")
print(fp32_output)

print("\nINT8 Output:")
print(int8_output)

difference = torch.mean(torch.abs(fp32_output - int8_output))

print(f"\nMean Absolute Difference: {difference.item():8f}")

quantized_path = "results/models/ddqn_dueling_seed42_quantized.pth"

torch.save(
    quantized_model.state_dict(),
    quantized_path,
)

quantized_size = Path(quantized_path).stat().st_size / 1024

compression_ratio = original_size / quantized_size

# --------------------------------------------------
# Latency Benchmark
# --------------------------------------------------

dummy_input = torch.randn(1, STATE_DIM)

N = 10000

with torch.no_grad():

    start = time.perf_counter()

    for _ in range(N):
        model(dummy_input)

    fp32_latency = (time.perf_counter() - start) / N * 1000

with torch.no_grad():

    start = time.perf_counter()

    for _ in range(N):
        quantized_model(dummy_input)

    int8_latency = (time.perf_counter() - start) / N * 1000

speedup = fp32_latency / int8_latency

# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 50)
print("DEPLOYMENT FEASIBILITY (DYNAMIC QUANTIZATION)")
print("=" * 50)

print(f"Original Model Size     : {original_size:.2f} KB")
print(f"Quantized Model Size    : {quantized_size:.2f} KB")
print(f"Compression Ratio       : {compression_ratio:.2f}x")

print()

print(f"FP32 Latency            : {fp32_latency:.6f} ms")
print(f"INT8 Latency            : {int8_latency:.6f} ms")
print(f"Inference Speedup       : {speedup:.2f}x")

print("=" * 50)