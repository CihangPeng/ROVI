# OVD Resampling

## Overview

This stage performs intelligent resampling of dense, overlapping detection results from multiple Open Vocabulary Detectors (OVDs) before [VLM cross-checking] (TBD). After OVD detection but before expensive VLM verification, this pipeline addresses the challenge of excessive box coverage by implementing importance-based sampling that compresses noisy results while preserving detection quality.
The core methodology utilizes combined views from different OVDs through voting mechanisms and overlap analysis to determine sampling priorities. By penalizing overlapping boxes, duplicate captions, distance from image center, and small box sizes, the resampling process effectively removes approximately 70% of instances while maintaining comprehensive image coverage. This strategic reduction significantly decreases computational overhead for the subsequent VLM cross-checking stage, transforming hundreds of dense detections into a manageable set of high-priority candidates for final verification.

## Requirements

- torch
- torchvision 
- supervision
- numpy
- pandas
- pillow

**Note**: You can easily add `supervision` and `pandas` to environments used during [vlm_description](../vlm_description) (part of the former stage of this pipeline).

## Usage

```bash
# Prepare your OVD results first
cd ./resampling

# Modify settings at the tops of ovd_resample.py and indexed_dataset.py

# Option 1: Run single process
python ovd_resample.py --divide_num 8 --data_slice_index 0

# Option 2: Run distributed processing using the launcher, modify run_resampling.sh first
chmod +x run_resampling.sh
./run_resampling.sh
```

**Warning**: In fact, this code might be challenging to run directly, as you'll need to build your own OVD results and VLM&LLM results data structure beforehand, with lots of dataset-related code updates required.

## Input/Output Structure

### Input Data Structure
The code expects the following input structure by default:

- **Image Data**: Raw image files in configured directories
- **OVD Results**: JSON files containing detection results from 4 OVD models:
  - `yw` (YOLO-World)
  - `ow` (OWLv2) 
  - `gd` (Grounding-DINO)
  - `od` (OV-DINO)
- **Caption Data**: VLM/LLM generated descriptions in JSON format

### Output Structure
- **JSON Metadata**: Detection results with sampling information
- **Annotated Images**: Visualizations with bounding boxes and labels
- **Process Logs**: Detailed sampling process information

## Configuration Settings

### Key Parameters in `get_config()`:

- **Stage Thresholds**: Control sampling behavior at each stage (e.g., `stage_3_expected_low_threshold`, `stage_4_single_low`)
- **Scoring Weights**: Balance different factors in selection (e.g., `stage_3_area_weight`, `stage_4_iou_weight`)
- **Image Processing**: Size constraints and transformations (`min_box_len`, `max_box_size`)
- **OVD Score Thresholds**: Per-model confidence thresholds (`yw_score_thr`, `ow_score_thr`, etc.)

## Resampling Strategy

The pipeline implements a 5-stage sampling strategy:

1. **Stage 0-1**: Preprocessing and filtering (score thresholds, area constraints, coordinate transforms)
2. **Stage 2**: Overlap removal and Non-Maximum Suppression (NMS) per OVD type
3. **Stage 3**: Equal contribution sampling to balance different OVD sources
4. **Stage 4**: OVD merge sampling with IoU-based selection and overlap punishment
5. **Stage 5**: Final selection with comprehensive overlap punishment and multi-replica generation

Each stage progressively refines the detection set, balancing quality, diversity, and avoiding redundancy across multiple OVD sources.

## Reference

For a comprehensive understanding of this resampling methodology, please refer to our [paper](TBD) and supplementary materials (which include detailed examples and ablation studies).
