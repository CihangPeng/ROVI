# VLM Description Stage

This pipeline stage provides batch image captioning using InternVL-Chat-V1.5 (26B parameters) for large-scale web data processing. It supports distributed multi-GPU processing with dynamic image resolution up to 4K.

## Prerequisites

### 1. Model Setup

By default, this script expects the InternVL-Chat-V1.5 model to be placed under `./InternVL-Chat-V1-5/` in the current directory.

**Download the model:**
```bash
# Using huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli download OpenGVLab/InternVL-Chat-V1-5 --local-dir ./InternVL-Chat-V1-5

# Or using git
git clone https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5
```

**Environment setup:**
- Please refer to the [InternVL-Chat-V1.5 HuggingFace page](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) for detailed environment setup instructions
- Ensure you have `transformers>=4.37.2`: `pip install transformers>=4.37.2`
- Required dependencies: `torch`, `torchvision`, `datasets`

### 2. Hardware Requirements

- **GPU Memory**: Requires approximately 48GB+ VRAM for the 26B model
- **Multi-GPU**: Supports distributed processing across multiple GPUs
- **Storage**: Sufficient disk space for outputs and logs

## Data Format

This code is designed for mass web data processing. At the time of our work, we store images as byte strings and package them into PyArrow format using the HuggingFace `datasets` library.

**Expected dataset structure:**
```python
{
    'new_key': str,        # Unique identifier
    'png': bytes,          # Image data as byte string
    'caption': str,        # Original web caption (for storage format only)
    # ... other fields
}
```

**Note on 'caption' field:** This stage does not require existing captions of images for processing. The 'caption' field is used only for creating a stable storage format that preserves original metadata alongside the new VLM-generated descriptions. Consider removing this field if it doesn't meet your data conditions.

**⚠️ Important:** You may need to modify the dataset-related code in `vlm_batch_processor.py` to suit your own data structure, particularly:
- The `collate_fn` function
- Data loading and processing logic
- Field names and data types

## Usage

### Setup (assume you've cloned the whole repo and are in the root directory)

```bash
cd vlm_description
```

### Basic Usage

```bash
# Run normally (foreground)
./unified_launcher.sh

# Run in background
./unified_launcher.sh &

# Stop specific GPU process
touch kill0.txt                               # Stop GPU 0
touch kill1.txt                               # Stop GPU 1

# Monitor progress
tail -f logs/gpu0.log                         # Monitor GPU 0
tail -f logs/*.log                            # Monitor all GPUs
```

### Configuration

Edit the configuration section in `unified_launcher.sh`:

```bash
# Parameters, adjust as needed
GPU_COUNT=1                                   # Number of GPUs to use
BSZ=6                                         # Batch size per GPU
MAX_NUM=9                                     # Max image tiles (1-40)
TEMPERATURE=0.6                               # Generation temperature
TEST_CUTDOWN=9999999                          # Limit samples for testing

# VLM Processing Configuration
SUB_PARTITION_NAME="part1"                    # Dataset partition name
SOURCE="url_undo_combined"                    # Dataset source name

# Paths, modified as needed
MODEL_PATH="./InternVL-Chat-V1-5"            # Model directory
DATASET_DIR="/path/to/your/dataset/"         # Dataset directory
OUTPUT_DIR="./outputs/out_${SOURCE}_${SUB_PARTITION_NAME}/"  # Output directory
LOG_DIR="./logs"                              # Log directory
```

## Output Structure

```
outputs/
└── out_${SOURCE}_${SUB_PARTITION_NAME}/
    ├── images/                               # Sample images (limited to 1000)
    ├── captions/                             # Generated captions (JSON)
    │   └── {key}.json                        # {"web_caption": "...", "vlm_caption": "..."}
    ├── queries/
    │   └── query.txt                         # Prompt template used
    └── end{gpu_id}.txt                       # Completion markers

logs/
├── gpu0.log                                  # GPU 0 processing log
├── gpu1.log                                  # GPU 1 processing log
└── ...
```

## Process Management

### Kill Signals
Create kill files to gracefully stop specific GPU processes:
```bash
touch kill0.txt    # Stop GPU 0
touch kill1.txt    # Stop GPU 1
touch kill2.txt    # Stop GPU 2
touch kill3.txt    # Stop GPU 3
```

### Monitoring
- **Logs**: Check `logs/gpu{id}.log` for detailed processing logs
- **Progress**: Monitor completion files `outputs/.../end{gpu_id}.txt`
- **Errors**: Check logs for error messages and troubleshooting

### Resuming
This stage automatically skips already processed items based on existing caption files. To resume after interruption, simply restart the script.

## Performance Notes

- **Batch Size**: Adjust `BSZ` based on your GPU memory (recommended: 4-8 for 80GB GPUs)
- **Image Tiles**: Higher `MAX_NUM` may provide better quality but slower processing (recommended: 6-12)
- **Multi-GPU**: Processing speed scales approximately linearly with GPU count
- **I/O**: Ensure fast storage for optimal performance with large datasets

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure transformers version >= 4.37.2 and complete model download
2. **GPU Memory Issues**: Reduce batch size (`BSZ`) or max tiles (`MAX_NUM`)
3. **Dataset Format**: Modify `collate_fn` and data loading code for your specific format
4. **Process Hanging**: Check logs for errors, use kill files for graceful shutdown

### Error Recovery
- Use kill files for clean shutdown
- Check individual GPU logs for specific errors
- Restart failed processes individually by modifying GPU_COUNT and data slicing

## Files

- `unified_launcher.sh`: Main launcher script
- `vlm_batch_processor.py`: Core processing logic
- `indexed_dataset.py`: Dataset slicing utility

For more details on the InternVL model capabilities and limitations, refer to the [official documentation](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5).
