#!/bin/bash

# --- Configuration ---
GPUS=4
CONFIG_FILE="configs/pointpillars/pointpillars_hv_fpn_sbn-all_1xb4-4e_nus-3d_MINI.py"

# --- Execution Command ---
echo "Starting distributed training on $GPUS GPUs with config: $CONFIG_FILE"
./tools/dist_train.sh "$CONFIG_FILE" "$GPUS"

echo "Training script finished."