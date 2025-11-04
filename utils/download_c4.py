# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# download_c4.py
import json
import os

from datasets import load_dataset

N_SAMPLES = int(os.getenv("N_SAMPLES", 10))
print(f" Downloading first {N_SAMPLES} samples from C4...")

# Use streaming to avoid downloading large files
ds = load_dataset("allenai/c4", name="en", split="train", streaming=True)

# Take only first N_SAMPLES
samples = list(ds.take(N_SAMPLES))

# Save as jsonl for TorchTitan compatibility
DATASET_BASE_PATH = os.getenv("DATASET_BASE_PATH", os.getcwd())
os.makedirs(DATASET_BASE_PATH, exist_ok=True)
with open(f"{DATASET_BASE_PATH}/train.jsonl", "w") as f:
    for sample in samples:
        json.dump(sample, f)
        f.write("\n")

print(f"Downloaded {len(samples)} samples")
print(f"Saved to: {DATASET_BASE_PATH}/train.jsonl")
