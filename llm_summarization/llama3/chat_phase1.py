# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
import fire
from llama import Dialog, Llama
from indexed_dataset import collate_fn, IndexedDataset
from torch.utils.data import DataLoader
import os
import json
import time
import re
from contextlib import contextmanager
from datetime import datetime

@contextmanager
def timeit_context(name="Task"):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{name} took {elapsed_time:.6f} seconds")

def get_system_guide(llama3_start):
    """Generate system instructions for attribute extraction."""
    return f"""\
You are precise, never imaging without any reference. \
You only speak the necessary words. \
Avoid repetitions and overlapping between attributes. \
DO NOT include notes or explanations. \
DO NOT output anything except the list mentioned. \
Combine separate adjectives with nouns according to the text. \
DO NOT output adjectives as independent attributes, as they shall always combined with nouns. \
Start with \"{llama3_start}\"\
"""

def get_query(web_caption, vlm_caption):
    """Generate query for attribute extraction from captions."""
    return f"""\
Given a fine-grained caption: \"{vlm_caption}\" \
And a noisy caption: \"{web_caption}\" \
Catch all the combinations, nouns, objects, and entities from the captions. \
Pay attention to any combinations in captions, \
and use combinations instead of single words if possible. \
Use adjective-noun combinations wherever possible \
to provide more specific and eloquent attributes. \
Extract potential combinations from sentences. \
Combine similar terms instead of individual types into combination wherever possible \
while still maintaining specificity if necessary. \
List the potential attributes, including words, phrases, and combinations. \
Use serial numbers to separate each attribute you are going to list.\
"""

def vlm_caption_preprocess(vlm_caption):
    """Preprocess VLM captions to remove common prefixes and clean text."""
    # Handle "This is" prefix
    is_capital_first = vlm_caption[0] == 'T' if vlm_caption else False
    str_to_replace = 'This is ' if is_capital_first else 'this is '
    vlm_caption = vlm_caption.replace(str_to_replace, '', 1)
    
    if is_capital_first and vlm_caption:
        vlm_caption = vlm_caption[0].upper() + vlm_caption[1:]
    
    # Remove trailing period
    if vlm_caption.endswith('.'):
        vlm_caption = vlm_caption[:-1]
    
    # Process sentences and filter unwanted content
    vlm_caption_split = vlm_caption.split('. ')
    processed_split = []
    
    unwanted_phrases = [
        "atmosphere", "lighting", "is no ", "is not ", "are no ", "are not "
    ]
    
    for item in vlm_caption_split:
        item = item.strip()
        if not item:
            continue
        
        # Skip sentences containing unwanted phrases
        if not any(phrase in item for phrase in unwanted_phrases):
            processed_split.append(item)
    
    return ". ".join(processed_split) + "." if processed_split else ""

def remove_parentheses(text):
    """Remove content within parentheses."""
    return re.sub(r'\([^()]*\)', '', text).strip()

def attr_postprocess(attributes, web_caption, vlm_caption):
    """Post-process extracted attributes to remove unwanted terms."""
    post_attrs = []
    unwanted_terms = [
        "atmosphere", "lighting", "background", "noisy caption",
        "fine-grained caption", "fine grained caption"
    ]
    
    # Add terms not found in source captions to unwanted list
    conditional_terms = ["fine-grained", "fine grained", "noisy"]
    for term in conditional_terms:
        if term not in web_caption and term not in vlm_caption:
            unwanted_terms.append(term)
    
    for attr in attributes:
        attr = attr.strip().lower().replace('_', ' ').replace(',', '')
        
        # Skip attributes containing unwanted terms
        if not any(term in attr for term in unwanted_terms):
            attr = remove_parentheses(attr)
            if attr:
                post_attrs.append(attr)
    
    return post_attrs

def get_dialog(batch, llama3_start):
    """Generate dialog batch for LLM processing."""
    web_caption_list = batch['web_caption_list']
    vlm_caption_list = batch['vlm_caption_list']
    assert len(web_caption_list) == len(vlm_caption_list)
    
    dialogs = []
    collections = []
    
    for web_caption, vlm_caption in zip(web_caption_list, vlm_caption_list):
        # Convert to lowercase for consistent processing
        web_caption = web_caption.lower()
        vlm_caption = vlm_caption.lower()
        
        processed_vlm_caption = vlm_caption_preprocess(vlm_caption)
        
        system_guide = get_system_guide(llama3_start)
        query = get_query(web_caption, processed_vlm_caption)
        
        collections.append({
            'processed_vlm_caption': processed_vlm_caption,
            'system': system_guide,
            'user': query
        })
        
        dialogs.append([
            {"role": "system", "content": system_guide},
            {"role": "user", "content": query}
        ])
        
    return dialogs, collections

def write_timestamp_to_file(file_path):
    """Write current timestamp to file."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a+') as file:
        file.write(current_time + '\n')

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 80,
    max_gen_len: Optional[int] = None,
    data_slice_index: int = 0,
    gpu_num: int = 1,
    dataset_filelist: str = "",
    root_dir: str = "./",
    output_dir: str = "./output"
):
    """Main processing function for Phase 1 attribute extraction."""
    
    # Create output directories
    output_json_dir = os.path.join(output_dir, 'json')
    output_dialog_dir = os.path.join(output_dir, 'dialog')
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_dialog_dir, exist_ok=True)
    
    # Parse dataset folders
    folders = dataset_filelist.split('[SEP]')
    print(f'Dataset folders: {folders}')
    
    # Initialize Llama model
    print(f'Loading model from {ckpt_dir}...')
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    # Create dataset with distributed slicing
    dataset = IndexedDataset(
        folders=folders,
        gpu_num=gpu_num,
        data_slice_index=data_slice_index
    )
    
    loader = DataLoader(
        dataset, 
        collate_fn=lambda x: collate_fn(x),
        batch_size=max_batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // gpu_num,
    )

    # Process control files
    # Note: kill files use GPU index (1-4), end files use data slice index + 1 (1-4)
    cuda_device_index = data_slice_index + 1  # This matches START_CUDA_INDEX=1 in shell scripts
    killing_file = os.path.join(root_dir, f'kill{cuda_device_index}.txt')
    ending_file = os.path.join(output_dir, f'end{cuda_device_index}.txt')
    
    dialog_saved_count = 0
    llama3_start = "Here is the list:"
    
    print(f'Starting Phase 1 processing for GPU {data_slice_index}...')
    
    while True:
        for ld_i, batch in enumerate(loader):
            # Check for kill signal
            if os.path.exists(killing_file):
                print(f'Kill signal detected for GPU {data_slice_index}, exiting...')
                os.remove(killing_file)
                return
            
            new_key_list = batch['new_key_list']
            web_caption_list = batch['web_caption_list']
            vlm_caption_list = batch['vlm_caption_list']
            
            dialogs, collections = get_dialog(batch, llama3_start)
            
            try:
                with timeit_context(f"Processing batch {ld_i}"):
                    results = generator.chat_completion(
                        dialogs,
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )

                # Save results if not already completed
                if not os.path.exists(ending_file):
                    for new_key, web_caption, vlm_caption, result, collection in zip(
                        new_key_list, web_caption_list, vlm_caption_list, results, collections
                    ):
                        # Process LLM output
                        processed_attrs = result['generation']['content']
                        original_answer = processed_attrs
                        
                        # Clean and parse attributes
                        processed_attrs = processed_attrs.replace(llama3_start, "").strip()
                        
                        # Remove trailing punctuation
                        while processed_attrs and processed_attrs[-1] in '.,\n ':
                            processed_attrs = processed_attrs[:-1]
                        
                        # Parse numbered list
                        attributes = []
                        for line in processed_attrs.split('\n'):
                            line = line.strip().replace('\n', '')
                            
                            # Remove numbering prefix
                            words = line.split(' ')
                            if len(words) > 1:
                                line = ' '.join(words[1:])
                            
                            if line and line not in attributes:
                                attributes.append(line)
                        
                        # Post-process attributes
                        attributes_postprocess = attr_postprocess(
                            attributes, vlm_caption, web_caption
                        )
                        
                        # Create output data
                        output_dict = {
                            'new_key': new_key,
                            'web_caption': web_caption,
                            'vlm_caption': vlm_caption,
                            'processed_vlm_caption': collection['processed_vlm_caption'],
                            'llama3_phase1_original_answer': original_answer,
                            'llama3_phase1_original_attributes': ', '.join(attributes),
                            'llama3_phase1_attributes': ', '.join(attributes_postprocess),
                        }
                        
                        # Save JSON output
                        output_path = os.path.join(output_json_dir, new_key + '.json')
                        with open(output_path, 'w') as f:
                            json.dump(output_dict, f, indent=4)
                        
                        # Save sample dialogs (first 500)
                        if dialog_saved_count <= 500:
                            dialog_path = os.path.join(output_dialog_dir, new_key + '.txt')
                            with open(dialog_path, 'w') as f:
                                for key, value in collection.items():
                                    f.write(f'{key}: {value}\n')
                                f.write(f'assistant: {original_answer}\n')
                            dialog_saved_count += 1
                        
                else:
                    print(f'End file exists: {ending_file}, skipping save')
                    
            except Exception as e:
                print(f'Exception occurred: {e}')
                print(f'Batch range: {new_key_list[0]} to {new_key_list[-1]}')
        
        # Mark completion
        if not os.path.exists(ending_file):
            write_timestamp_to_file(ending_file)
            print(f'Phase 1 processing completed for GPU {data_slice_index}')
            break

if __name__ == "__main__":
    fire.Fire(main)
