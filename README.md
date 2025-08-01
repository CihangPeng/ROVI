# ROVI: A VLM-LLM Re-Captioned Dataset for Open-Vocabulary Instance-Grounded Text-to-Image Generation

## Overview

ROVI is a dataset featuring 1M curated web images with comprehensive image descriptions and bounding box annotations. Using a novel VLM-LLM re-captioning strategy, ROVI exceeds existing detection-centric datasets in image description, quality, and resolution, while containing two orders of magnitude more categories with an open-vocabulary nature. Our approach yields a global prompt inherently linked to instance annotations while capturing secondary visual elements humans typically overlook.

This work is accepted at ICCV 2025. For demonstrative purposes, instance-grounded T2I generators trained on ROVI achieve superior performance in grounding accuracy, prompt fidelity, and aesthetic quality.

**Paper:** Coming soon

**Dataset:** [ROVI Dataset on Hugging Face](https://huggingface.co/datasets/CHang/ROVI)

**Demo:** [ROVI Dataset Example Viewer](https://huggingface.co/spaces/CHang/ROVI-Dataset-Example-Viewer) - This demo will fetch a random 1k subset from ROVI val set, and display up to 100 random images with rich annotations.

## Dataset Structure

The dataset is provided in JSON format with train and validation splits:

- **Train Set:** 981,551 samples (keys: '0000001' - '0981551')
- **Validation Set:** 30,153 samples (keys: '0981552' - '1011704')

Each sample is stored as a dictionary with a 7-digit key and contains the following fields:

### Core Fields

- **`url`**: Image URL
- **`source`**: Data source with quality filtering
  - `laion_aes`: From [LAION-5B](https://laion.ai/blog/laion-5b/) with aesthetic score ≥ 6.0
  - `coyo_6plus`: From [COYO-700M](https://github.com/kakaobrain/coyo-dataset) with aesthetic score ≥ 6.0  
  - `coyo_add`: From [COYO-700M](https://github.com/kakaobrain/coyo-dataset) with aesthetic score 5.75-6.0
  - `laion_pop`: From [LAION-POP](https://laion.ai/blog/laion-pop/) for diversity (high average aesthetic score)
- **`width`**, **`height`**: Image dimensions
- **`box_num`**: Number of bounding boxes
- **`category_num`**: Number of distinct categories in the sample

### Caption Fields

- **`web_caption`**: Original caption from source metadata (LAION-5B, COYO-700M, or LAION-POP)
- **`vlm_description`**: Generated description using [InternVL-1.5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5) (as described in paper pipeline)

### Tokenization Information

- **`web_clip_tok_num`**, **`vlm_clip_tok_num`**: Token counts based on CLIP tokenizer
- **`web_clip_last_sentence_idx`**, **`vlm_clip_last_sentence_idx`**: Character index for complete last sentence when using 77-token CLIP limit

### Metadata

- **`phash`**: Perceptual hash for deduplication
- **`source_meta`**: Unique source information inherited from original metadata

### Annotations

All annotation fields (`labels`, `bboxes`, `scores`, `ovd_belongings`) are lists of the same length.

- **`labels`**: Open-vocabulary object labels (strings)
- **`bboxes`**: Bounding box coordinates (xyxy format)
- **`scores`**: Detection confidence scores (may vary across detectors)
- **`ovd_belongings`**: Detection source attribution
  - `gd`: [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO)
  - `yw`: [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
  - `ow`: [OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
  - `od`: [OV-DINO](https://github.com/wanghao9610/OV-DINO)

## Pipeline Stages

Coming soon...

## Related Repositories

This project builds upon several key repositories:

- **InternVL-1.5**: [OpenGVLab/InternVL-Chat-V1-5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)
- **LLaMA-3**: [meta-llama/llama3](https://github.com/meta-llama/llama3)
- **Qwen2-VL**: [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- **Grounding-DINO**: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- **YOLO-World**: [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- **OWLv2**: [Hugging Face OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
- **OV-DINO**: [wanghao9610/OV-DINO](https://github.com/wanghao9610/OV-DINO)

## License

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Limitations

- Image availability may change over time as URLs become inaccessible
- Detection annotations are generated automatically and may contain errors
- Minor language model artifacts persist, including inconsistent singular/plural handling and awkward phrasing
- Bounding box grounding may be less accurate for visually occluded objects and non-contiguous elements

For more detailed discussion, please refer to our paper and supplementary materials.

## Acknowledgments

We thank the LAION-5B and COYO-700M programs for providing the foundational image datasets, and the authors and contributors of InternVL-1.5, LLaMA-3, Qwen2-VL, Grounding-DINO, YOLO-World, OWLv2, and OV-DINO for their outstanding work and contributions to the open-source community, which made this dataset possible.

## Contact

For questions or issues regarding this dataset, please contact: cihangpeng@zju.edu.cn
