#!/bin/bash

# Phase 2: Attribute Refinement and Merging
# Processes Phase 1 outputs to refine and merge attributes

NAME="attribute_refinement_phase2"

# Input from Phase 1 - modify to your Phase 1 output directory
dataset_filelist="/path/to/llama3/phase1/output/attribute_extraction_phase1/json/"

output_dir_root="/path/to/llama3/phase2/output/"

# Processing Parameters
NUM_DEVICE=4                               # Number of GPUs to use
START_CUDA_INDEX=1                         # Starting GPU index (0-based)

# Model Parameters - refined settings for attribute processing
max_seq_len=1024                          # Maximum sequence length
max_batch_size=100                        # Batch size per GPU (can be higher than Phase 1)
temperature=0.4                           # Lower temperature for more focused processing
top_p=0.5                                 # Lower top-p for more deterministic results

# Path Configuration - should match Phase 1 setup
root_dir="/path/to/llama3/"
cuda_sh_path="./llama3/multi_cuda.sh"
python_path="./llama3/chat_phase2.py"
ckpt_dir="./Meta-Llama-3-8B-Instruct/"
tokenizer_path="./Meta-Llama-3-8B-Instruct/tokenizer.model"

# Ensure output directory exists
output_dir_root=$(echo "$output_dir_root" | sed 's:/*$::')
name_as="${NAME}"

if [ -d "$output_dir_root" ]; then
    echo "Directory '$output_dir_root' already exists."
else
    mkdir -p "$output_dir_root"
    echo "Directory '$output_dir_root' created."
fi

output_dir="${output_dir_root}/${name_as}"
if [ -d "$output_dir" ]; then
    echo "Directory '$output_dir' already exists."
else
    mkdir -p "$output_dir"
    echo "Directory '$output_dir' created."
fi

# Backup scripts to output directory for reproducibility
cp "$python_path" "$output_dir"
cp "$0" "$output_dir"
cp "$cuda_sh_path" "$output_dir"

# Launch distributed processing across GPUs
echo "Starting Phase 2 attribute refinement on $NUM_DEVICE GPUs..."

i=0
while [ $i -le $((NUM_DEVICE - 1)) ]
do
    cuda_device_index=$((i + START_CUDA_INDEX))
    data_slice_index=$i

    echo "Launching GPU ${cuda_device_index}, dataset slice ${data_slice_index}"
    sh "$cuda_sh_path" \
        "$cuda_device_index" \
        "$data_slice_index" \
        "$NUM_DEVICE" \
        "$python_path" \
        "$ckpt_dir" \
        "$tokenizer_path" \
        "$dataset_filelist" \
        "$root_dir" \
        "$output_dir" \
        "$max_seq_len" \
        "$max_batch_size" \
        "$temperature" \
        "$top_p" \
        > "${output_dir}/cuda_${cuda_device_index}.txt" &
    
    i=$((i + 1))
done

echo "Phase 2 processes launched. Monitor progress with:"
echo "  tail -f ${output_dir}/cuda_*.txt"
echo "Stop individual processes with:"
echo "  touch kill{1,2,3,4}.txt"