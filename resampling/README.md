# OVD Resampling

## Overview

This stage performs intelligent resampling of dense, overlapping detection results from multiple Open Vocabulary Detectors (OVDs) before [VLM cross-checking](../vlm_cross_checking). After OVD detection but before expensive VLM verification, this pipeline addresses the challenge of excessive box coverage by implementing importance-based sampling that compresses noisy results while preserving detection quality.

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
  - `yw` ([YOLO-World](https://github.com/AILab-CVC/YOLO-World))
  - `ow` ([OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2)) 
  - `gd` ([Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO))
  - `od` ([OV-DINO](https://github.com/wanghao9610/OV-DINO))
- **Caption Data**: VLM/LLM generated descriptions in JSON format

### Output Structure
- **JSON Metadata**: Detection results with sampling information
- **Annotated Images**: Visualizations with bounding boxes and labels
- **Process Logs**: Detailed sampling process information

## Configuration Settings

### Key Parameters in [get_config()](ovd_resample.py#L70):

- **Stage Thresholds**: Control sampling behavior at each stage (e.g., `stage_3_expected_low_threshold`, `stage_4_single_low`)
- **Scoring Weights**: Balance different factors in selection (e.g., `stage_3_area_weight`, `stage_4_iou_weight`)
- **Image Processing**: Size constraints and transformations (`min_box_len`, `max_box_size`)
- **OVD Score Thresholds**: Per-model confidence thresholds (`yw_score_thr`, `ow_score_thr`, etc.)

## Resampling Strategy

The pipeline implements a 5-stage sampling strategy:

1. **Stage 0-1**: Preprocessing and filtering (score thresholds, area constraints, coordinate transforms)
2. **Stage 2**: Overlap removal and Non-Maximum Suppression (NMS) per OVD type
3. **Stage 3**: Adaptive sampling to harmonize detection contributions across OVD sources
4. **Stage 4**: OVD merge sampling with IoU-based selection and overlap punishment
5. **Stage 5**: Final selection with comprehensive overlap punishment and multi-replica generation

Each stage progressively refines the detection set, balancing quality, diversity, and avoiding redundancy across multiple OVD sources.

## Reference

For a comprehensive understanding of this resampling methodology, please refer to our [paper](TBD) and supplementary materials (which include detailed examples and ablation studies).

## Citation

This implementation adapts components from OpenMMLab MMDetection:

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

**Code Reference**: The `BaseDataElement` and `InstanceData` classes in `ovd_utils.py` are adapted from [MMDetection's data structures](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/data_structures/instance_data.py).
