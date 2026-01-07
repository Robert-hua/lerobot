#!/bin/bash

# Merge pick_bottle datasets script
# This script merges pick_bottle_1, pick_bottle_long, and pick_bottle_max_speed
# into a single dataset called pick_bottle_all

set -e  # Exit on error

# Configuration
# Auto-detect data root or use specified path
if [ -d "/mnt/data/data_hub" ]; then
    DATA_ROOT="/mnt/data/data_hub"
elif [ -d "/workspace/data_hub" ]; then
    DATA_ROOT="/workspace/data_hub"
else
    echo "ERROR: Cannot find data_hub directory!"
    echo "Please set DATA_ROOT manually in the script."
    exit 1
fi

OUTPUT_REPO_ID="pick_bottle_all"

# Dataset repo IDs to merge (these should match the actual dataset names in DATA_ROOT)
DATASETS=(
    "pick_bottle_1"
    "pick_bottle_long"
    "pick_bottle_max_speed"
)

echo "=========================================="
echo "Merging Pick Bottle Datasets"
echo "=========================================="
echo "Data root: ${DATA_ROOT}"
echo "Output dataset: ${OUTPUT_REPO_ID}"
echo "Datasets to merge:"
for dataset in "${DATASETS[@]}"; do
    echo "  - ${dataset}"
done
echo "=========================================="

# Check if datasets exist
echo ""
echo "Checking if datasets exist..."
MISSING_DATASETS=()
for dataset in "${DATASETS[@]}"; do
    if [ ! -d "${DATA_ROOT}/${dataset}" ]; then
        echo "  ✗ ${dataset} NOT FOUND at ${DATA_ROOT}/${dataset}"
        MISSING_DATASETS+=("${dataset}")
    else
        echo "  ✓ ${dataset} found"
    fi
done

if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: ${#MISSING_DATASETS[@]} dataset(s) not found!"
    echo "Missing datasets: ${MISSING_DATASETS[*]}"
    echo ""
    echo "Available datasets in ${DATA_ROOT}:"
    ls -1 "${DATA_ROOT}/" 2>/dev/null || echo "  (directory not accessible)"
    echo ""
    echo "Please check:"
    echo "  1. Dataset names are correct"
    echo "  2. Datasets exist in ${DATA_ROOT}"
    echo "  3. You have read permissions"
    exit 1
fi

# Run the merge operation
echo ""
echo "Starting merge operation..."
python3 -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id "${OUTPUT_REPO_ID}" \
    --root "${DATA_ROOT}" \
    --operation.type merge \
    --operation.repo_ids "[\"pick_bottle_1\", \"pick_bottle_long\", \"pick_bottle_max_speed\"]"

# Check if merge was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Merge completed successfully!"
    echo "Output dataset: ${DATA_ROOT}/${OUTPUT_REPO_ID}"
    echo "=========================================="
    
    # Show dataset info
    if [ -d "${DATA_ROOT}/${OUTPUT_REPO_ID}" ]; then
        echo ""
        echo "Dataset contents:"
        ls -lh "${DATA_ROOT}/${OUTPUT_REPO_ID}/"
    fi
else
    echo ""
    echo "=========================================="
    echo "✗ Merge failed!"
    echo "=========================================="
    exit 1
fi
