#!/usr/bin/bash
set -e
set -x

bash reset_neuron_cores_memory_peaks.sh

# Neuron Runtime Configuration
: ${NEURON_RT_VIRTUAL_CORE_SIZE:=2}
: ${NEURON_RT_NUM_CORES:=4}
export NEURON_RT_VIRTUAL_CORE_SIZE NEURON_RT_NUM_CORES

# Torch Neuron Configuration
: ${TORCH_NEURON_SYNC_MODE:=1}
: ${NEURON_LAUNCH_BLOCKING:=0}
: ${TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS:=0}
export TORCH_NEURON_SYNC_MODE NEURON_LAUNCH_BLOCKING TORCH_NEURON_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS

# Training Configuration
: ${CONFIG_FILE:="./torchtitan/models/qwen3/train_configs/qwen3_8b_tp4.toml"}
TRAIN_FILE="torchtitan.train"

# MXFP8 Configuration
if [ "${NEURON_ADDMM_MXFP8:-0}" = "1" ]; then
    echo "NEURON_ADDMM_MXFP8 is SET"
    export NEURON_RT_ENABLE_OCP_SATURATION=1
    export NEURON_RT_ENABLE_OCP=1
fi

export LOG_NAME="test_run_log"
torchrun --nnodes 1 --nproc_per_node $NEURON_RT_NUM_CORES --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "localhost:29500" -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@" 2>&1 | tee "${LOG_NAME}.log"

mkdir -p $LOG_NAME
export NEURON_MEMORY_STATS_DUMP_DIR=$LOG_NAME && ./get_neuron_peak_memory.sh
