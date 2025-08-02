#!/bin/bash

# Parameters, adjust as needed
GPU_COUNT=1                                   # Number of GPUs to use
BSZ=6                                         # Batch size per GPU
MAX_NUM=9                                     # Max image tiles (1-40)
TEMPERATURE=0.6                               # Generation temperature
TEST_CUTDOWN=9999999                          # Limit samples for testing

# Dataset Configuration
SUB_PARTITION_NAME="your_partition"           # Dataset partition name
SOURCE="your_data_source"                     # Dataset source name

# Paths, modified as needed
MODEL_PATH="./InternVL-Chat-V1-5"            # Model directory
DATASET_DIR="/path/to/your/dataset/"         # Dataset directory
OUTPUT_DIR="./outputs/out_${SOURCE}_${SUB_PARTITION_NAME}/"  # Output directory
LOG_DIR="./logs"                              # Log directory

# Script location
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PYTHON_SCRIPT="$SCRIPT_DIR/vlm_batch_processor.py"

# Validation
[[ -f "$PYTHON_SCRIPT" ]] || { echo "Error: $PYTHON_SCRIPT not found"; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "Error: Model not found at $MODEL_PATH"; exit 1; }
[[ -d "$DATASET_DIR" ]] || { echo "Error: Dataset not found at $DATASET_DIR"; exit 1; }

# Create directories
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Processing: $SOURCE/$SUB_PARTITION_NAME on $GPU_COUNT GPUs"
echo "Output: $OUTPUT_DIR"

# Launch processes
for gpu in $(seq 0 $((GPU_COUNT-1))); do
    CUDA_VISIBLE_DEVICES=$gpu python "$PYTHON_SCRIPT" \
        --dataset_dir "$DATASET_DIR" \
        --root_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH" \
        --kill_index $gpu \
        --bsz $BSZ \
        --max_num $MAX_NUM \
        --temperature $TEMPERATURE \
        --gpu_num $GPU_COUNT \
        --data_slice_index $gpu \
        --test_cutdown $TEST_CUTDOWN \
        > "$LOG_DIR/gpu${gpu}.log" 2>&1 &
    
    echo "GPU $gpu: PID $!"
done

echo "Use 'touch kill{0..$((GPU_COUNT-1))}.txt' to stop processes"
wait
