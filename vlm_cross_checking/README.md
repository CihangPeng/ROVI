# VLM Cross-Checking Stage

This pipeline stage provides batch object detection verification using [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) for large-scale bounding box validation. It processes cropped image regions and determines whether detected objects match their predicted labels.

## Prerequisites

### 1. Model Setup

```bash
cd vlm_cross_checking
```

First, refer to the [Qwen2-VL-7B-Instruct HuggingFace page](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) to set up your environment with required dependencies like `transformers` and `torch`.

Then download the model using our script:

```bash
python download_qwen2vl.py
```

This will download the model to `./huggingface_model/` directory structure.

**Requirements:**
- GPU Memory: Minimum 16GB VRAM for 7B model

### 2. Hardware Requirements

- **GPU Memory**: Minimum 16GB VRAM for the 7B model
- **Multi-GPU**: Supports distributed processing across multiple GPUs
- **CPU & RAM**: High-frequency image preprocessing with cropping operations requires sufficient RAM and CPU cores for optimal efficiency

## Input Data Format

**Important**: You will likely need to modify the dataset loading code in `indexed_dataset.py` to match your own data organization, file paths, and naming conventions. The current implementation assumes this specific mapping structure.

This **stage of the pipeline** uses a **mapping system** to locate images and their corresponding OVD annotation files from previous pipeline stages.

**Our Current Mapping Structure:**

After the [resampling stage](../resampling), we have a structure where **one image is paired with multiple OVD detection results**. This follows the **resampling approach** mentioned in our paper (like peeling layers of an onion) - each image gets several sampled detection results for verification.

Our mapping files work as follows:

1. **`img_dirs.json`** - Maps image identifiers to their storage directories:
```json
{
    "image_id_1": "/path/to/directory1/",
    "image_id_2": "/path/to/directory2/",
    "...": "..."
}
```

2. **`todo_img_ovd_map.json`** - Maps each image to its multiple OVD annotation files:
```json
{
    "image_id_1": {
        "0": "/path/to/ovd_annotation_1.json",
        "1": "/path/to/ovd_annotation_2.json",
        "2": "/path/to/ovd_annotation_3.json"
    },
    "...": "..."
}
```

3. **OVD annotation files** contain bounding boxes and detected objects:
```javascript
{
    "new_key": "image_identifier",
    "OV_merged": ["object1", "object2", "..."],
    "bboxes": [[x0, y0, x1, y1], [x0, y0, x1, y1], "..."],
    "...": "..."
}
```

## Usage

### Setup

```bash
# 1. Download model (one-time setup)
python download_qwen2vl.py

# 2. Prepare required mapping files, or modify indexed_dataset.py for your data structure:
#    - img_dirs.json (maps image IDs to directories)
#    - todo_img_ovd_map.json (maps images to OVD annotations)

# 3. Run verification
chmod +x run_qwen2vl.sh
./run_qwen2vl.sh

# 4. Run in background
./run_qwen2vl.sh &
```

### Configuration

Edit parameters in `run_qwen2vl.sh`:

```bash
GPU_COUNT=4                   # Number of GPUs to use (default: 4)
IMG_BSZ=24                    # Image batch size per GPU
BOX_BSZ=40                    # Box batch size per GPU
MAX_LEN=392                   # Maximum crop size
SECONDARY_MAX_LEN=224         # Secondary max size
MIN_LEN=28                    # Minimum crop size
```

### Process Management

```bash
# Stop specific GPU process
touch output/*/kill0.txt     # Stop GPU 0
touch output/*/kill1.txt     # Stop GPU 1
# ... etc for other GPUs

# Check completion
ls output/*/end*.txt         # Check which GPUs completed

# Single GPU mode (edit run_qwen2vl.sh)
# Comment out multi-GPU section, uncomment single GPU section
```

## File Structure

```
vlm_cross_checking/
├── cross_checking.py                 # Main processing script
├── indexed_dataset.py                # Dataset handling
├── download_qwen2vl.py               # Model download script
├── run_qwen2vl.sh                    # Execution script
├── README.md                         # This file
│
├── img_dirs.json                     # Image directory mappings (REQUIRED)
├── todo_img_ovd_map.json             # Image-OVD mappings (REQUIRED)
│
├── huggingface_model/                # Model directory (created by download_qwen2vl.py)
│   ├── hf_home/                      # HuggingFace cache
│   ├── model/                        # Qwen2-VL model files
│   └── processor/                    # Processor files
│
└── output/                           # Processing outputs (created by run_qwen2vl.sh)
    └── {timestamp}_{source}_{name}/
        ├── vlm_check_update/         # Verification results
        ├── kill{gpu_id}.txt          # Kill signals
        └── end{gpu_id}.txt           # Completion markers
```

## Output Format

Verification results are saved as JSON files:

```javascript
{
    "new_key": "image_identifier",
    "sub_sample": 0,
    "image_size": [width, height],
    "bdx_list": [0, 1, 2],
    "ov_list": ["object1", "object2", "object3"],
    "original_judge_list": ["yes", "no", "yes"],
    "modified_judge_list": ["yes", "no", "no"],
    "judge_probs_list": ["(0.8234_0.1766)", "(0.2341_0.7659)", "(0.5123_0.4877)"],
    "box_list": [[x0, y0, x1, y1], ...],
    "query_list": ["Is this an image of object1? Answer yes or no.", ...],
    "answer_list": ["Yes", "No", "Yes"]
}
```

### Yes/No Decision Making

The verification system uses a **one-token prediction approach** with probability-based refinement:

**Token Analysis**: The model predicts the next token after the prompt "Is this an image of {object}? Answer yes or no." The system analyzes probabilities for all variations of yes/no tokens:
- **Yes tokens**: `Yes`, `yes`, `YES`
- **No tokens**: `No`, `no`, `NO`

**Probability Calculation**: For each prediction, we sum the probabilities of all yes tokens and all no tokens separately. The `judge_probs_list` shows these sums as `(yes_prob_sum_no_prob_sum)`.

**Decision Logic**:
- **`original_judge_list`**: Raw model output parsing - if the generated text starts with "yes"/"Yes"/"YES" or "no"/"No"/"NO", otherwise marked as "err"
- **`modified_judge_list`**: **Refined decision using stricter criteria** to address VLM yes-bias observed in our statistical analysis:

```python
# Error correction for unparseable responses
if original_judge == 'err':
    if yes_prob < 0.3: → 'no'
    elif yes_prob >= no_prob + 0.2: → 'yes' 
    else: → 'no'

# Bias correction for "yes" responses  
elif original_judge == 'yes':
    if yes_prob < no_prob + 0.2: → 'no'  # Stricter threshold
```

**Rationale**: Based on empirical observation, VLMs exhibit a **tendency toward positive responses** in yes/no questions. The `modified_judge_list` applies stricter probability thresholds to reduce false positives, requiring stronger evidence (≥0.2 probability margin) to maintain a "yes" classification.

For detailed model information, visit the [Qwen2-VL-7B-Instruct HuggingFace page](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).
