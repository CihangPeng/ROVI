import os
import json
import time
import argparse
from datetime import datetime
from contextlib import contextmanager

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.data import DataLoader

from indexed_dataset import OvdDataset, img_collate_fn


@contextmanager
def timeit_context(name="Task"):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{name} took {elapsed_time:.6f} seconds")


def write_timestamp_to_file(file_path):
    """Write current timestamp to file."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a+') as file:
        file.write(current_time + '\n')
        file.seek(0)
        print(file.read())


def kill_signal(killing_file):
    """Check for kill signal and exit if found."""
    if os.path.exists(killing_file):
        print(f'Found kill file: {killing_file}, exiting')
        os.remove(killing_file)
        exit()


def main():
    # Environment setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--cuda_device_index", type=int, required=True, help="CUDA device index")
    parser.add_argument("--img_bsz", type=int, default=24, help="Image batch size")
    parser.add_argument("--box_bsz", type=int, default=40, help="Box batch size")
    parser.add_argument("--max_len", type=int, default=392, help="Maximum crop size")
    parser.add_argument("--secondary_max_len", type=int, default=224, help="Secondary maximum size")
    parser.add_argument("--min_len", type=int, default=28, help="Minimum crop size")
    
    args = parser.parse_args()
    
    print(f'GPU {args.cuda_device_index} | img_bsz:{args.img_bsz} | box_bsz:{args.box_bsz}')
    print(f'max_len:{args.max_len} | secondary_max_len:{args.secondary_max_len} | min_len:{args.min_len}')
    
    # Setup directories
    os.makedirs(args.root_dir, exist_ok=True)
    
    # Setup kill/end files
    ending_file = os.path.join(args.root_dir, f'end{args.cuda_device_index}.txt')
    killing_file = os.path.join(args.root_dir, f'kill{args.cuda_device_index}.txt')
    
    # Load model and processor
    model_dir = "./huggingface_model/model"
    processor_dir = "./huggingface_model/processor"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found. Please run 'python download.py' first.")
    if not os.path.exists(processor_dir):
        raise FileNotFoundError(f"Processor directory {processor_dir} not found. Please run 'python download.py' first.")
    
    processor = AutoProcessor.from_pretrained(processor_dir)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto"
    ).cuda()
    
    # Setup yes/no token mappings for verification
    yes_no_str_list = ['Yes', 'yes', 'YES', 'No', 'no', 'NO']
    yes_no_decode = [processor.tokenizer.encode(yes_no_str) for yes_no_str in yes_no_str_list]
    
    for key, decode_k in zip(yes_no_str_list, yes_no_decode):
        assert len(decode_k) == 1, f"{key} has more than 1 token: {decode_k}"
    
    yes_no_tok_list = [processor.tokenizer.convert_tokens_to_ids(yn_s) for yn_s in yes_no_str_list]
    yes_tok_list = yes_no_tok_list[:len(yes_no_tok_list) // 2]
    no_tok_list = yes_no_tok_list[len(yes_no_tok_list) // 2:]
    
    print('Valid yes|no tokens:', yes_no_tok_list)
    
    # Create dataset and dataloader
    img_dataset = OvdDataset(
        box_bsz=args.box_bsz,
        max_len=args.max_len,
        min_len=args.min_len,
        secondary_max_len=args.secondary_max_len,
    )
    
    img_loader = DataLoader(
        img_dataset, 
        collate_fn=img_collate_fn,
        batch_size=args.img_bsz, 
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    
    print(f'Image loader created with {len(img_loader)} batches')
    
    # Main processing loop
    while True:
        kill_signal(killing_file)
        
        for box_loader in img_loader:
            kill_signal(killing_file)
            
            print(f'Processing box loader with {len(box_loader)} batches')
            
            # Collect results for all boxes in this image batch
            results = {
                'box_belonging_list': [],
                'bdx_list': [],
                'ov_list': [],
                'img_size_list': [],
                'box_list': [],
                'query_list': [],
                'answer_list': [],
                'judge_probs_list': [],
                'original_judge_list': [],
                'modified_judge_list': [],
                'merged_ovd_path_list': [],
            }
            
            for batch in box_loader:
                kill_signal(killing_file)
                
                # Prepare messages and images for batch inference
                messages = []
                images = []
                
                for idx, ov in enumerate(batch['ov_str']):
                    cropped_img = batch['cropped_img_list'][idx]
                    
                    messages.append([{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"Is this an image of {ov}? Answer yes or no."},
                        ],
                    }])
                    images.append(cropped_img)
                
                # Process batch
                texts = [
                    processor.apply_chat_template(msg, add_generation_prompt=True)
                    for msg in messages
                ]
                
                inputs = processor(
                    text=texts, images=images, padding=True, return_tensors="pt"
                ).to("cuda")
                
                # Inference
                with timeit_context(f"Inference batch size {len(messages)}"), torch.no_grad():
                    outputs = model(**inputs, use_cache=False)
                    logits = outputs.logits[:, -1, :]
                    output_ids = logits.argmax(dim=-1)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                
                generated_ids = output_ids.cpu().tolist()
                
                # Calculate yes/no probabilities
                judge_probs = [
                    (
                        sum([prob[yes_tok].item() for yes_tok in yes_tok_list]),
                        sum([prob[no_tok].item() for no_tok in no_tok_list]),
                    ) for prob in probs
                ]
                
                output_texts = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                # Process results
                if not os.path.exists(ending_file):
                    for idx, output_text in enumerate(output_texts):
                        query = texts[idx]
                        answer = output_text
                        judge_prob = judge_probs[idx]
                        
                        # Parse original judgment
                        ori_judge = (answer.strip())[:3].lower()
                        if ori_judge == 'yes':
                            pass
                        else:
                            ori_judge = ori_judge[:2]
                            if ori_judge == 'no':
                                pass
                            else:
                                ori_judge = 'err'
                        
                        # Apply probability-based correction
                        modified_judge = ori_judge
                        if ori_judge == 'err':
                            if judge_prob[0] < 0.3:
                                modified_judge = 'no'
                            elif judge_prob[0] >= judge_prob[1] + 0.2:
                                modified_judge = 'yes'
                            else:
                                modified_judge = 'no'
                        elif ori_judge == 'yes':
                            if judge_prob[0] < judge_prob[1] + 0.2:
                                modified_judge = 'no'
                        
                        new_key = batch['new_key_list'][idx]
                        sub_sample = batch['sub_sample_list'][idx]
                        bdx = batch['bdx_list'][idx]
                        ov = batch['ov_str'][idx]
                        
                        judge_prob_str = f'({judge_prob[0]:.4f}_{judge_prob[1]:.4f})'
                        
                        # Store results
                        results['merged_ovd_path_list'].append(batch['merged_ovd_path_list'][idx])
                        results['box_belonging_list'].append(f'{new_key}_{sub_sample}')
                        results['bdx_list'].append(bdx)
                        results['ov_list'].append(ov)
                        results['answer_list'].append(answer)
                        results['original_judge_list'].append(ori_judge)
                        results['modified_judge_list'].append(modified_judge)
                        results['judge_probs_list'].append(judge_prob_str)
                        results['img_size_list'].append(batch['img_size_list'][idx])
                        results['box_list'].append(batch['box_list'][idx])
                        results['query_list'].append(query)
                
                del batch
            
            # Save results grouped by image
            set_box_belonging = set(results['box_belonging_list'])
            
            for box_belonging in set_box_belonging:
                idx_list = [i for i, x in enumerate(results['box_belonging_list']) if x == box_belonging]
                
                # Get metadata (same for all boxes from same image)
                img_size = results['img_size_list'][idx_list[0]]
                merged_ovd_path = results['merged_ovd_path_list'][idx_list[0]]
                
                slice_dir = os.path.dirname(os.path.dirname(merged_ovd_path))
                new_key = box_belonging[:-2]
                sub_sample = int(box_belonging[-1])
                
                out_dict = {
                    "new_key": new_key,
                    "sub_sample": sub_sample,
                    "image_size": img_size,
                    'bdx_list': [results['bdx_list'][i] for i in idx_list],
                    'ov_list': [results['ov_list'][i] for i in idx_list],
                    'judge_probs_list': [results['judge_probs_list'][i] for i in idx_list],
                    'modified_judge_list': [results['modified_judge_list'][i] for i in idx_list],
                    'original_judge_list': [results['original_judge_list'][i] for i in idx_list],
                    'query_list': [results['query_list'][i] for i in idx_list],
                    'answer_list': [results['answer_list'][i] for i in idx_list],
                    'box_list': [results['box_list'][i] for i in idx_list],
                }
                
                # Save to output directory
                vlm_check_dir = os.path.join(slice_dir, 'vlm_check_update')
                os.makedirs(vlm_check_dir, exist_ok=True)
                vlm_check_path = os.path.join(vlm_check_dir, box_belonging + '.json')
                
                with open(vlm_check_path, 'w') as f:
                    json.dump(out_dict, f, indent=4)
            
            del box_loader
        
        # Mark completion
        if not os.path.exists(ending_file):
            write_timestamp_to_file(ending_file)


if __name__ == "__main__":
    main()