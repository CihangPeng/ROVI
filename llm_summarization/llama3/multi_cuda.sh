#!/bin/bash

# Multi-GPU coordinator script for distributed LLM processing
# Handles individual GPU process execution with proper port management

CUDA_INDEX="$1"
data_slice_index="$2"
gpu_num="$3"
python_path="$4"
ckpt_dir="$5"
tokenizer_path="$6"
dataset_filelist="$7"
root_dir="$8"
output_dir="$9"
max_seq_len="${10}"
max_batch_size="${11}"
temperature="${12}"
top_p="${13}"
max_gen_len="None"

echo "=== GPU $CUDA_INDEX Process Configuration ==="
echo "CUDA_INDEX: $CUDA_INDEX"
echo "data_slice_index: $data_slice_index"
echo "gpu_num: $gpu_num"
echo "python_path: $python_path"
echo "ckpt_dir: $ckpt_dir"
echo "dataset_filelist: $dataset_filelist"
echo "output_dir: $output_dir"
echo "max_batch_size: $max_batch_size"
echo "temperature: $temperature"
echo "============================================="

script_dir=$(dirname "$python_path")
cd "$script_dir" || exit

# Dynamic port allocation to avoid conflicts
BASE_PORT=12340
PORT=$((BASE_PORT + CUDA_INDEX))

echo "Using port $PORT for GPU $CUDA_INDEX"

# Launch distributed training with torchrun
CUDA_VISIBLE_DEVICES=$CUDA_INDEX torchrun \
    --rdzv_backend=static \
    --nproc_per_node 1 \
    --master_port=$PORT \
    "$python_path" \
    --ckpt_dir "$ckpt_dir" \
    --tokenizer_path "$tokenizer_path" \
    --temperature $temperature \
    --top_p $top_p \
    --max_seq_len $max_seq_len \
    --max_batch_size $max_batch_size \
    --max_gen_len $max_gen_len \
    --data_slice_index $data_slice_index \
    --gpu_num $gpu_num \
    --dataset_filelist "$dataset_filelist" \
    --root_dir "$root_dir" \
    --output_dir "$output_dir"