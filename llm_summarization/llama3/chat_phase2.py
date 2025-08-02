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
import random
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

def get_system_guide(new_elements_start, discarded_attributes_start):
    """Generate system instructions for attribute refinement."""
    return f"""\
You are precise, never imaging without any reference. \
You only speak the necessary words. \
Skip the attributes if they do not perfectly match one of the rules. \
The new elements list shall only contain nouns or noun phrases, \
while adjectives are prohibited. \
Avoid repetitions and overlapping, \
especially between the split new elements and the old attributes. \
DO NOT include notes or explanations. \
DO NOT output anything except the list mentioned. \
DO NOT output adjectives as independent attributes, as they shall always combined with nouns. \
DO NOT copy and paste the attributes without the conditions mentioned. \
Start the list of new elements with \"{new_elements_start}\" \
If the list contains nothing, you shall only output a newline character after the start.\
"""

def get_query(phase1_attributes):
    """Generate query for attribute refinement."""
    return f"""\
Given a list of attributes: \"{phase1_attributes}\", separate with commas. \
You shall output some new elements according to the following rules. \
Think step by step. \
Firstly, check the attributes one by one. \
Skip all the nouns or noun phrases. \
Secondly, group the attributes that use highly overlapping word combinations. \
For each group, \
extract only the different parts of nouns or noun phrases from similar combinations as new elements. \
Thirdly, split the bare nouns from combined adjective-noun phrases as new elements, \
especially for long phrases that have more than two words. \
Split singular nouns or noun phrases from plural forms as new elements. \
The bare nouns typically include one or two words. \
At last, remove all the colors in new elements. \
List the new elements. \
Use serial numbers to separate each attribute you are going to list.\
"""

def attr_postprocess(attributes):
    """Post-process extracted attributes."""
    post_attrs = []
    unwanted_terms = [
        "atmosphere", "lighting", "background", "noisy caption",
        "fine-grained caption", "fine grained caption"
    ]
    
    for attr in attributes:
        attr = attr.strip().lower().replace('_', ' ')
        
        # Skip attributes with unwanted terms
        if not any(term in attr for term in unwanted_terms):
            if attr:
                post_attrs.append(attr)
    
    return post_attrs

def symbol_mapping(phase1_list):
    """Create mapping for hyphenated terms."""
    phase_keys = []
    phase_values = []
    for phase1_attr in phase1_list:
        phase_values.append(phase1_attr)
        phase_keys.append(phase1_attr.replace('-', ' '))
    
    return dict(zip(phase_keys, phase_values))

def attr_list_cmp(attr_list1, attr_list2):
    """Compare attribute lists for similarity."""
    attr_list1_set = set(attr_list1)
    attr_list2_set = set(attr_list2)
    
    common_words = attr_list1_set.intersection(attr_list2_set)
    
    # Check if attributes have significant overlap
    return (len(common_words) >= 3 or 
            (float(len(common_words)) / len(attr_list1_set) >= 0.65 and
             float(len(common_words)) / len(attr_list2_set) >= 0.65))

def attr_final_merge(phase2_list, phase1_str):
    """Merge Phase 2 results with Phase 1 attributes."""
    phase1_list = [attr.strip() for attr in phase1_str.split(',') if attr.strip()]
    
    # Create symbol mapping for hyphenated terms
    symbol_mapping_dict = symbol_mapping(phase1_list)
    
    # Split attributes into words (max 4 words)
    phase1_list_split_words = [
        phase1_attr.replace('-', ' ').split(' ') 
        for phase1_attr in phase1_list
    ]
    phase1_list_split_no_more_than_4 = [
        attr_split for attr_split in phase1_list_split_words 
        if len(attr_split) <= 4
    ]
    
    random.shuffle(phase1_list_split_no_more_than_4)
    
    # Remove highly similar attributes
    phase1_list_split_remain = []
    for i, phase1_attr_split in enumerate(phase1_list_split_no_more_than_4):
        is_unique = True
        for j in range(i + 1, len(phase1_list_split_no_more_than_4)):
            other_attr = phase1_list_split_no_more_than_4[j]
            if attr_list_cmp(phase1_attr_split, other_attr):
                is_unique = False
                break
        if is_unique:
            phase1_list_split_remain.append(phase1_attr_split)
    
    # Map back to original format with hyphens
    phase1_list_remain = [
        symbol_mapping_dict[' '.join(attr_split)] 
        for attr_split in phase1_list_split_remain
    ]
    
    # Union phase2 and remaining phase1 attributes
    phase2_set = set(phase2_list)
    phase1_remain_set = set(phase1_list_remain)
    result_list = list(phase1_remain_set.union(phase2_set))
    
    # Handle singular/plural forms
    for i, item in enumerate(result_list):
        item = item.strip()
        if item and item.endswith('s') and item[:-1] in result_list:
            result_list[i] = ""
    
    result_list = [item.strip() for item in result_list if item.strip()]
    
    # Human term normalization
    human_mapping = {
        'men': 'man', 'women': 'woman', 
        'children': 'child', 'people': 'person'
    }
    result_list = [
        human_mapping.get(item, item) for item in result_list
    ]
    result_list = list(set(result_list))
    
    # Add human terms if found in compound phrases
    def is_valid_in(word, str_to_check):
        return (str_to_check.startswith(word + ' ') or
                str_to_check.endswith(' ' + word) or
                f' {word} ' in str_to_check)
    
    for plural, singular in human_mapping.items():
        if singular not in result_list:
            for item in result_list:
                if is_valid_in(plural, item) or is_valid_in(singular, item):
                    result_list.append(singular)
                    break
    
    # Sort by length
    result_list.sort(key=len)
    
    return result_list

def get_dialog(batch, new_elements_start, discarded_attributes_start):
    """Generate dialog batch for Phase 2 processing."""
    assert 'llama3_phase1_attributes_list' in batch.keys()
    phase1_attributes_list = batch['llama3_phase1_attributes_list']
    
    dialogs = []
    collections = []
    
    for phase1_attributes in phase1_attributes_list:
        phase1_attributes = phase1_attributes.lower()
        
        system_guide = get_system_guide(new_elements_start, discarded_attributes_start)
        query = get_query(phase1_attributes)
        
        collections.append({
            'phase1_attributes': phase1_attributes,
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
    temperature: float = 0.4,
    top_p: float = 0.5,
    max_seq_len: int = 1024,
    max_batch_size: int = 100,
    max_gen_len: Optional[int] = None,
    data_slice_index: int = 0,
    gpu_num: int = 1,
    dataset_filelist: str = "",
    root_dir: str = "./",
    output_dir: str = "./output"
):
    """Main processing function for Phase 2 attribute refinement."""
    
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
    new_elements_start = 'Here is the list of new elements:'
    discarded_attributes_start = 'Here is the list of best combinations:'
    
    print(f'Starting Phase 2 processing for GPU {data_slice_index}...')
    
    while True:
        for ld_i, batch in enumerate(loader):
            # Check for kill signal
            if os.path.exists(killing_file):
                print(f'Kill signal detected for GPU {data_slice_index}, exiting...')
                os.remove(killing_file)
                return
            
            dialogs, collections = get_dialog(batch, new_elements_start, discarded_attributes_start)
            new_key_list = batch['new_key_list']
            
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
                    for sample, result, collection in zip(
                        batch['sample_list'], results, collections
                    ):
                        new_key = sample['new_key']
                        
                        # Process LLM output
                        processed_attrs = result['generation']['content']
                        original_answer = processed_attrs
                        
                        # Clean and parse attributes
                        processed_attrs = processed_attrs.replace(new_elements_start, "").strip()
                        
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
                        
                        # Post-process and merge attributes
                        attributes_postprocess = attr_postprocess(attributes)
                        attributes_merge = attr_final_merge(
                            attributes_postprocess, sample['llama3_phase1_attributes']
                        )
                        
                        # Update sample with Phase 2 results
                        sample['llama3_phase2_original_answer'] = original_answer
                        sample['llama3_phase2_original_attributes'] = ', '.join(attributes)
                        sample['llama3_phase2_attributes'] = ', '.join(attributes_postprocess)
                        sample['llama3_phase2_final_merge'] = ', '.join(attributes_merge)
                        
                        # Save JSON output
                        output_path = os.path.join(output_json_dir, new_key + '.json')
                        with open(output_path, 'w') as f:
                            json.dump(sample, f, indent=4)
                        
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
            print(f'Phase 2 processing completed for GPU {data_slice_index}')
            break

if __name__ == "__main__":
    fire.Fire(main)