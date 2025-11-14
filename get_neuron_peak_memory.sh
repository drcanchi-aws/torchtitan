#!/bin/bash
set -e

# Function to read peak memory for a specific neuron device and core
read_peak_memory() {
    local device=$1
    local core=$2
    local base_path="/sys/devices/virtual/neuron_device/neuron${device}/neuron_core${core}/stats/memory_usage/device_mem"
    local categories=("collectives" "constants" "dma_rings" "driver_memory" "model_code" "model_shared_scratchpad" "nonshared_scratchpad" "notifications" "runtime_memory" "tensors" "uncategorized")
    local values=()

    # Function to convert bytes to GB with 3 decimal places
    bytes_to_gb() {
        echo "scale=3; $1 / 1024 / 1024 / 1024" | bc
    }

    # Collect all category values
    for category in "${categories[@]}"; do
        local peak=$(sudo cat "${base_path}/${category}/peak")
        local peak_gb=$(bytes_to_gb $peak)
        values+=("$peak_gb")
    done

    # Add total peak
    local total_peak=$(sudo cat "${base_path}/peak")
    local total_peak_gb=$(bytes_to_gb $total_peak)
    values+=("$total_peak_gb")

    # Print row to CSV
    printf "neuron%d-core%d," $device $core >&3
    printf "%.3f," "${values[@]}" >&3
    printf "\n" >&3

    # Return values for max calculation (without printing)
    echo "${values[@]}"
}

echo "Reading peak memory usage files..."

# Initialize arrays for storing maximum values
declare -a max_values
categories=("collectives" "constants" "dma_rings" "driver_memory" "model_code" "model_shared_scratchpad" "nonshared_scratchpad" "notifications" "runtime_memory" "tensors" "uncategorized" "Total_Peak")
for i in "${!categories[@]}"; do
    max_values[$i]=0
done

# Set filepath to cwd if NEURON_MEMORY_STATS_DUMP_DIR is not set
filepath="${NEURON_MEMORY_STATS_DUMP_DIR:-$(pwd)}/memory_log.csv"
exec 3>$filepath
printf "Device/Core," >&3
printf "%s," "${categories[@]}" >&3
printf "\n" >&3

# Read data for all devices and cores and track maximum values
for device in {0..15}; do
    for core in {0..7}; do
        if [ -d "/sys/devices/virtual/neuron_device/neuron${device}/neuron_core${core}" ]; then
            # Read values and update maximums
            read -r -a current_values <<< $(read_peak_memory $device $core)
            for i in "${!current_values[@]}"; do
                if (( $(echo "${current_values[$i]} > ${max_values[$i]}" | bc -l) )); then
                    max_values[$i]=${current_values[$i]}
                fi
            done
        fi
    done
done

# Print MAX row
printf "MAX," >&3
for value in "${max_values[@]}"; do
    printf "%.3f," "$value" >&3
done
printf "\n" >&3

# Close file descriptor
exec 3>&-

# Print maximum values to terminal
echo -e "\nMaximum Peak Memory Usage Summary:"
echo "-----------------------------------"
for i in "${!categories[@]}"; do
    printf "%-25s %.3f GB\n" "${categories[$i]}" "${max_values[$i]}"
done
echo "-----------------------------------"

echo "Peak memory usage collection complete! Detailed breakdown across ranks has been saved to $filepath"