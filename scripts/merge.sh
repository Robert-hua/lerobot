#!/bin/bash

# Merge pick_bottle datasets script
# This script merges pick_bottle_1, pick_bottle_long, and pick_bottle_max_speed
# into a single dataset called pick_bottle_all

set -e  # Exit on error

# Configuration
DATA_ROOT="/workspace/data_hub"
OUTPUT_REPO_ID="pick_bottle_all"

# Dataset paths to merge
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
for dataset in "${DATASETS[@]}"; do
    if [ ! -d "${DATA_ROOT}/${dataset}" ]; then
        echo "ERROR: Dataset ${DATA_ROOT}/${dataset} does not exist!"
        exit 1
    fi
    echo "  ✓ ${dataset} found"
done

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

