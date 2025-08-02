#!/bin/bash

# Configuration
SUB_PARTITION_NAME="qwen2_verification"
SOURCE="vlm_check"

# Batch sizes and image processing parameters
IMG_BSZ=24
BOX_BSZ=40
MAX_LEN=392
SECONDARY_MAX_LEN=224
MIN_LEN=28

# Output directory
TIME=$(date +%m%d%H%M)
ROOT_DIR="./output/${TIME}_${SOURCE}_${SUB_PARTITION_NAME}/"

# Create output directory
mkdir -p "$ROOT_DIR"

echo "Starting VLM verification pipeline..."
echo "Output directory: $ROOT_DIR"
echo "Image batch size: $IMG_BSZ"
echo "Box batch size: $BOX_BSZ"

# Check required files
if [ ! -f "img_dirs.json" ]; then
    echo "Error: img_dirs.json not found. Please ensure your data files are set up."
    exit 1
fi

if [ ! -f "todo_img_ovd_map.json" ]; then
    echo "Error: todo_img_ovd_map.json not found. Please ensure your data files are set up."
    exit 1
fi

# Check model files
if [ ! -d "huggingface_model/model" ]; then
    echo "Error: Model not found. Please run 'python download_qwen2vl.py' first."
    exit 1
fi

# Multi-GPU execution (default)
GPU_COUNT=4

echo "Starting ${GPU_COUNT} GPU processes..."

for i in $(seq 0 $((GPU_COUNT-1))); do
    echo "Starting GPU $i process..."
    CUDA_VISIBLE_DEVICES=$i python cross_checking.py \
        --root_dir "$ROOT_DIR" \
        --cuda_device_index $i \
        --img_bsz $IMG_BSZ \
        --box_bsz $BOX_BSZ \
        --max_len $MAX_LEN \
        --secondary_max_len $SECONDARY_MAX_LEN \
        --min_len $MIN_LEN &
done

echo "All processes started in background"
echo "To stop GPU i: touch ${ROOT_DIR}kill{i}.txt"
echo "To monitor: check ${ROOT_DIR}end{i}.txt for completion"