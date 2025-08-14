# %%

import time
import numpy as np
import torch

# %%
# Simulated dataset
data = np.random.rand(1000000)

def cpu_sum(data):
    start = time.time()
    result = sum(data)
    end = time.time()
    print(f"CPU sum: {result:.2f}, Time: {end - start:.4f}s")

def numpy_sum(data):
    start = time.time()
    result = np.sum(data)
    end = time.time()
    print(f"NumPy sum: {result:.2f}, Time: {end - start:.4f}s")

try:
    def gpu_sum(data):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tensor = torch.tensor(data, device=device)
        start = time.time()
        result = torch.sum(tensor).item()
        end = time.time()
        print(f"PyTorch ({device}) sum: {result:.2f}, Time: {end - start:.4f}s")
except ImportError:
    def gpu_sum(data):
        print("PyTorch not installed, skipping GPU example.")

print("Running sum on different runtimes/hardware:")
cpu_sum(data)
numpy_sum(data)
gpu_sum(data)
# %%
