# download_c4.py
from datasets import load_dataset, Dataset
import os
import json

N_SAMPLES = int(os.getenv("N_SAMPLES", 10))
print(f" Downloading first {N_SAMPLES} samples from C4...")

# Use streaming to avoid downloading large files
ds = load_dataset(
    'allenai/c4', 
    name='en', 
    split='train', 
    streaming=True
)

# Take only first N_SAMPLES
samples = list(ds.take(N_SAMPLES))

# Save as jsonl for TorchTitan compatibility
base_dir = f"./datasets/c4_local_{N_SAMPLES}_samples"
os.makedirs(base_dir, exist_ok=True)

with open(f'{base_dir}/train.jsonl', 'w') as f:
    for sample in samples:
        json.dump(sample, f)
        f.write('\n')

print(f"Downloaded {len(samples)} samples")
print(f"Saved to: ./datasets/c4_local/train.jsonl")