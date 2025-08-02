# LLM Summarization Stage

This pipeline stage provides two-phase attribute processing using Llama3-8B-Instruct for large-scale image caption processing. It processes VLM-generated captions to extract structured attributes in two phases: initial compound phrase extraction and decomposition into constituent parts.

## Prerequisites

### 1. Model Setup

**Llama3 Preparation:**
```bash
cd llm_summarization/

# Backup our custom scripts
mkdir -p llama3_backup/
cp llama3/*.py llama3/*.sh llama3_backup/

# Remove illustration folder and clone official repo
rm -rf llama3/
git clone https://github.com/meta-llama/llama3.git

# run download script
./download_llama3.sh
# Select option for Meta-Llama-3-8B-Instruct model

# Restore our custom scripts and cleanup
cp llama3_backup/* llama3/
rm -rf llama3_backup/
```

**Expected Final Structure:**
```
llm_summarization/
├── README.md
├── run_llama3_phase1.sh
├── run_llama3_phase2.sh
├── download_llama3.sh              # Official download script
├── Meta-Llama-3-8B-Instruct/       # Downloaded model files
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   ├── params.json
│   └── tokenizer.model
└── llama3/                          # Official Llama3 repository + our scripts
    ├── llama/                       # Official Llama3 core modules
    │   ├── __init__.py
    │   ├── model.py
    │   ├── tokenizer.py
    │   └── generation.py
    ├── indexed_dataset.py           # Our dataset utilities
    ├── chat_phase1.py               # Our Phase 1 processing script
    ├── chat_phase2.py               # Our Phase 2 processing script
    ├── multi_cuda.sh                # Our multi-GPU coordinator
    └── requirements.txt             # Official requirements
```

**Important Setup Note:** The `llama3/` folder shown in this repository is for illustration purposes only. You'll need to:
1. Temporarily move our scripts out of the `llama3/` folder
2. Clone the official Llama3 repository 
3. Place our modified scripts back into the cloned repository

**Environment setup:**
- Follow Llama3 official installation requirements from the cloned repository
- Required dependencies: `torch`, `fire`, `datasets` (install via `pip install -r llama3/requirements.txt`)
- Ensure GPU compatibility for distributed processing

### 2. Hardware Requirements

- **GPU Memory**: Requires approximately 16GB+ VRAM per GPU for 8B model
- **Multi-GPU**: Supports distributed processing across multiple GPUs
- **Storage**: Sufficient disk space for outputs and logs

## Data Format

This stage processes VLM caption outputs from the previous stage. Expected input format:

```python
{
    'new_key': str,           # Unique identifier
    'web_caption': str,       # Original web caption
    'vlm_caption': str,       # VLM-generated description
    # ... other fields
}
```

## Pipeline Stages

### Phase 1: Initial Attribute Extraction
Extracts structured attributes and compound phrases from web captions and VLM descriptions.

### Phase 2: Attribute Decomposition
Breaks down complex phrases into constituent parts (e.g., extracting basic nouns from compound phrases) to improve detection coverage.

## Usage

### Setup

```bash
cd llm_summarization
```

### Configuration

Edit the configuration sections in the shell scripts:

**Phase 1 Configuration (`run_llama3_phase1.sh`):**
```bash
# Dataset paths - modify according to your VLM output structure
dataset_filelist="\
/path/to/vlm/output1/captions/[SEP]\
/path/to/vlm/output2/captions/[SEP]\
..."

# Processing parameters
NUM_DEVICE=4                               # Number of GPUs to use
START_CUDA_INDEX=1                         # Starting GPU index
max_seq_len=1024                          # Maximum sequence length
max_batch_size=80                         # Batch size per GPU
temperature=0.6                           # Generation temperature
top_p=0.7                                 # Top-p sampling

# Model paths
ckpt_dir="/path/to/Meta-Llama-3-8B-Instruct/"
tokenizer_path="/path/to/Meta-Llama-3-8B-Instruct/tokenizer.model"
output_dir_root="/path/to/phase1/output/"
```

**Phase 2 Configuration (`run_llama3_phase2.sh`):**
```bash
# Input from Phase 1 output
dataset_filelist="/path/to/phase1/output/json/"

# Processing parameters (typically smaller batch size for refinement)
max_batch_size=100
temperature=0.4                           # Lower temperature for more focused refinement
top_p=0.5
```

### Execution

**Run Phase 1:**
```bash
./run_llama3_phase1.sh
```

**Monitor Phase 1:**
```bash
tail -f /path/to/phase1/output/cuda_*.txt
```

**Run Phase 2 (after Phase 1 completion):**
```bash
./run_llama3_phase2.sh
```

### Process Management

**Stop specific GPU processes:**
```bash
touch kill1.txt    # Stop GPU 1 process
touch kill2.txt    # Stop GPU 2 process
```

**Check completion:**
Look for `end{gpu_id}.txt` files in output directories.

## Output Structure

**Phase 1 Output:**
```
phase1_output/
├── json/                                 # Extracted attributes
│   └── {key}.json                        # {"new_key": "...", "llama3_phase1_attributes": "..."}
├── dialog/                               # Sample dialogs (first 500)
│   └── {key}.txt
└── end{gpu_id}.txt                       # Completion markers
```

**Phase 2 Output:**
```
phase2_output/
├── json/                                 # Refined attributes
│   └── {key}.json                        # Final merged attributes
├── dialog/                               # Sample dialogs
└── end{gpu_id}.txt                       # Completion markers
```

## Processing Pipeline Details

### Phase 1: Attribute Extraction
- Input: Web captions + VLM descriptions
- Process: Extract nouns, objects, entities, and attribute combinations
- Output: Initial structured attribute lists
- Focus: Comprehensive attribute capture with adjective-noun combinations

### Phase 2: Attribute Refinement
- Input: Phase 1 attributes
- Process: Group similar attributes, split complex phrases, remove redundancies
- Output: Clean, merged attribute lists
- Focus: Quality improvement and deduplication

## Troubleshooting

### Common Issues

1. **Model Loading**: Ensure proper Llama3 model download and file structures
2. **GPU Memory**: Reduce batch sizes if encountering OOM errors
3. **Path Issues**: Verify all paths in configuration match your setup
4. **Process Hanging**: Use kill files for graceful shutdown, check logs for errors

### Error Recovery
- Use kill files for clean shutdown
- Check individual GPU logs in output directories
- Restart failed processes by modifying GPU configurations

For Llama3 model details and licensing, refer to the [official Meta Llama repository](https://github.com/meta-llama/llama3).
