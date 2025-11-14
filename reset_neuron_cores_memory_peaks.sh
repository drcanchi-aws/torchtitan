#!/bin/bash

# Function to write zero to all peak memory files
write_zero_to_peaks() {
    local device=$1
    local core=$2
    local base_path="/sys/devices/virtual/neuron_device/neuron${device}/neuron_core${core}/stats/memory_usage/device_mem"
    local categories=("collectives" "constants" "dma_rings" "driver_memory" "model_code" "model_shared_scratchpad" "nonshared_scratchpad" "notifications" "runtime_memory" "tensors" "uncategorized")

    # echo "Zeroing memory peaks for neuron${device}-core${core}..."

    for category in "${categories[@]}"; do
        sudo bash -c "echo 0 > ${base_path}/${category}/peak"
    done
    sudo bash -c "echo 0 > ${base_path}/peak"
}
# Zero out peaks for all devices and cores
echo "Reset peak memory info on all devices..."
for device in {0..15}; do
    for core in {0..7}; do
        if [ -d "/sys/devices/virtual/neuron_device/neuron${device}/neuron_core${core}" ]; then
            write_zero_to_peaks $device $core
        fi
    done
done
echo "Finished resetting memory info!"
echo "Finished Neuron Environment Setup!"