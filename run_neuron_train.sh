#!/usr/bin/bash
set -e
set -x

# SLABHS SETTING FP8 ENV VARS
export NEURON_RT_ENABLE_OCP_SATURATION=1
export NEURON_RT_ENABLE_OCP=1


# Set Neuron environment variables
export NEURON_RT_VIRTUAL_CORE_SIZE=2
export NEURON_RT_NUM_CORES=4
export TORCH_NEURON_SYNC_MODE=1
export NEURON_LAUNCH_BLOCKING=0

# Boolean mask indexing not supported on Neuron, falling back to CPU
export TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0

# Verbose RT logs
# export NEURON_RT_LOG_LEVEL_NRT="DEBUG"

# Configuration
CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_8b_reduce_size_tp4.toml"
# CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_8b_tp4.toml"
TRAIN_FILE="torchtitan.train"

# Execute training with srun
# srun --export=ALL torchrun --nnodes 1 --nproc_per_node 1 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "localhost:29500" -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} 2>&1 | tee ./qwen_8b.log

# Register ADDMM NKI MXFP8 / XLA IMPL
export NEURON_ADDMM_MXFP8=1

torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "localhost:29500" -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@" 2>&1 | tee ./test.log