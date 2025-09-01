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
import concurrent.futures
import os
# %%

def chunked_sum(chunk):
    """
    Calculates the sum of all elements in the given chunk (iterable).
    Args: chunk (iterable): An iterable of numeric values to be summed.
    Returns: numeric: The sum of all elements in the chunk.
    """
    return sum(chunk)

# Below is a function that performs parallel summation of a large dataset using 
# multiple processes.
# It divides the dataset into chunks, computes the sum of each chunk in parallel,
# and then combines the results to get the total sum.
def parallel_sum(data, max_workers=None):
    # Split data into chunks for each worker
    n_workers = max_workers or os.cpu_count()
    chunk_size = len(data) // n_workers
    # below is a list comprehension to create chunks
    #they have three parts: expression, iterable, and condition
    # The expression is the item to be included in the new list.
    # The iterable is the data to be processed, and the condition is optional.
    # Syntax: [expression for item in iterable if condition]
    # This creates a list of chunks, where each chunk is a slice of the data.
    # [expression for item in iterable "if condition"]
    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(n_workers)]
    # Handle any leftover data
    if len(data) % n_workers:
        chunks[-1] = np.concatenate([chunks[-1], data[n_workers*chunk_size:]])
    
    start = time.time() # Start timing the parallel computation
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(chunked_sum, chunks)) # Creates a pool of worker processes (up to max_workers), 
        #allowing tasks to run in parallel. results :Distributes each chunk to a worker process, 
        # where chunked_sum computes the sum of each chunk. 
    total = sum(results) #adds up the partial sums returned by each worker
    end = time.time() # records the time after the computation is done
    # Prints the total sum and the time taken for the parallel computation
    print(f"Parallel sum ({n_workers} workers): {total:.2f}, Time: {end - start:.4f}s")

# %%
print("\nParallel vs single-core sum:")
# Single core (no parallelism)
parallel_sum(data, max_workers=1)
# All available cores
parallel_sum(data, max_workers=os.cpu_count())

# %%
# How would we check for 2 and 3 cores? 