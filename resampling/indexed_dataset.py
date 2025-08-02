# ============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE FOR YOUR SETUP
# ============================================================================

# Data source paths - Update these paths according to your data location
IMG_DATA_FOLDERS = [
    "/your/img/dir1",
    "/your/img/dir2",
]

# OVD detection result paths - Update these paths for your OVD model outputs
OVD_DICT = {
    'yw': [
        "/your/yw/anno/dir1",
        "/your/yw/anno/dir2",
    ],
    'ow': [
        "/your/ow/anno/dir1",
        "/your/ow/anno/dir2",
    ],
    'gd': [
        "/your/gd/anno/dir1",
        "/your/gd/anno/dir2",
    ],
    'od': [
        "/your/od/anno/dir1",
        "/your/od/anno/dir2",
    ]
}

# Caption data paths
LLAMA3_JSON_FOLDERS = ["/your/llm/summarization/done"]
CAPTION_KEY_STR = 'llama3_json_file'

# Dataset size - Total number of samples expected
TOTAL_DATASET_SIZE = 1000000 # shall modify to match your actual number

# ============================================================================


import os
import torch
import pandas as pd
import copy
import json
import numpy as np
from PIL import Image, ImageFile
import io
import time
from contextlib import contextmanager

ImageFile.LOAD_TRUNCATED_IMAGES = True


@contextmanager
def timeit_context(name="Task", silent=False, log_significant=True):
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        if not silent and (log_significant and elapsed_time > 2.0):
            print(f"{name} took {elapsed_time:.6f} seconds")


def process_types_of_img_bytes(image_data):
    """Process various image formats and handle transparency/color modes.
    
    Args:
        image_data (bytes): Raw image bytes
        
    Returns:
        PIL.Image: Processed RGB image
    """
    image = Image.open(io.BytesIO(image_data))
    
    if image.mode in ('RGBA', 'LA') \
    or image.mode == 'P' and 'transparency' in image.info \
    or image.mode in ('RGBa', 'La') \
    or image.mode == 'CMYK':
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')
    
    if image.mode == 'RGBA':
        img = np.array(image)
        alpha = img[:, :, 3, np.newaxis]
        img = alpha / 255 * img[..., :3] + 255 - alpha
        img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)
        image = Image.fromarray(img, 'RGB')
        
    return image


def collect_json_or_img_files(folders, ext=None):
    """Collect files from multiple folders with optional extension filter.
    
    Args:
        folders (list): List of folder paths
        ext (str, optional): File extension filter
        
    Returns:
        list: List of file paths
    """
    find_files = []
    
    for folder in folders:
        if os.path.isdir(folder):
            print(f'Loading dataset from {folder}')
            
            if ext is not None:
                find_files.extend([
                    os.path.join(folder, filename) \
                    for filename in os.listdir(folder) \
                    if filename.endswith(ext)
                ])
            else:
                find_files.extend([
                    os.path.join(folder, filename) \
                    for filename in os.listdir(folder)
                ])
            
            file_type = ext if ext is not None else 'bytestring'
            print(f'{file_type} dataset loaded: {len(find_files)} files')
        else:
            print(f'Dataset directory does not exist: {folder}')
            exit()
            
    return find_files


def read_single_json_file(merged_dict, json_file, name_as_str, folder_list, add_prefix=None, time_cost_check=True):
    """Read and merge JSON file data into existing dictionary.
    
    Args:
        merged_dict (dict): Existing dictionary to merge into
        json_file (str): Path to JSON file to read
        name_as_str (str): Key name for storing file path
        folder_list (list): List of folders to search for JSON file
        add_prefix (str, optional): Prefix to add to all keys from JSON
        time_cost_check (bool): Whether to enable timing checks
        
    Returns:
        dict: Updated merged dictionary
    """
    rank = os.environ.get('LOCAL_RANK', '0')
    
    with timeit_context(f"{rank} | check key 1"):
        assert name_as_str not in merged_dict.keys(), \
            f'ERROR: key {name_as_str} already exists in merged_dict'
        new_key = merged_dict['new_key']
    
    if json_file is None:
        with timeit_context(f"{rank} | find existing path"):
            for json_folder in folder_list:
                expected_json_file = os.path.join(json_folder, new_key + '.json')
                if os.path.exists(expected_json_file):
                    json_file = expected_json_file
                    break
            if json_file is None:
                print(f'JSON file not found for {new_key}')
                exit()
    
    with timeit_context(f"{rank} | file open & json load"):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        assert json_data['new_key'] == new_key, \
            f'new_key {json_data["new_key"]} is wrong in {json_file}'
    
    with timeit_context(f"{rank} | pop new_key"):
        json_data.pop('new_key')
    
    with timeit_context(f"{rank} | check key 2"):    
        for key in json_data.keys():
            assert key not in merged_dict.keys(), \
                f'ERROR: reading {name_as_str}, but key {key} already exists in merged_dict'
    
    with timeit_context(f"{rank} | add prefix & new bind"):
        if add_prefix is not None:
            json_data = {add_prefix + key: json_data[key] for key in json_data}
    
    with timeit_context(f"{rank} | merge & new bind"):
        if merged_dict is not None:
            new_dict = {**copy.deepcopy(merged_dict), **json_data}
            new_dict[name_as_str] = json_file
            return new_dict
        else:
            return json_data


class MultiSubsetWrapper(torch.utils.data.Dataset):
    """Dataset wrapper for handling multiple data subsets with OVD annotations.
    
    Manages image data, captions, and object detection results from multiple sources.
    Supports distributed processing and configurable data slicing.
    """
    
    def __init__(self, config, subset_idx, start_index=0, end_index=9999999, cut_down=9999999):
        """Initialize the dataset wrapper.
        
        Args:
            config: Configuration object with dataset parameters
            subset_idx: Index of the current subset for distributed processing
            start_index: Starting index for data slice
            end_index: Ending index for data slice
            cut_down: Maximum number of samples to include
        """
        self.config = copy.deepcopy(config)
        self.subset_idx = subset_idx
        self.start_index = start_index
        
        assert end_index > start_index
        self.end_index = min(end_index, start_index + cut_down)
        
        self._setup_data_paths()
        self._build_dataset()
        
    def _setup_data_paths(self):
        """Configure paths for different data sources."""
        self.img_data_folders = IMG_DATA_FOLDERS
        self.ovd_dict = OVD_DICT
        self.llama3_json_folders = LLAMA3_JSON_FOLDERS
        self.caption_key_str = CAPTION_KEY_STR
        
    def _build_dataset(self):
        """Build the complete dataset from all sources."""
        self.all_img_filepaths = []
        self.all_ovd_dict_filepaths = {ovd_type: [] for ovd_type in self.ovd_dict.keys()}
        self.all_caption_filepaths = []
        self.cumulative_lengths = [0]
        
        # Collect files from all image data folders
        for idx, img_folder in enumerate(self.img_data_folders):
            img_subset = collect_json_or_img_files([img_folder], ext=None)
            
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(img_subset))
            self.all_img_filepaths.extend(img_subset)
            
            # Add corresponding caption files
            caption_path = self.llama3_json_folders[0]
            caption_subset = [
                os.path.join(caption_path, os.path.basename(img_file) + '.json') 
                for img_file in img_subset
            ]
            self.all_caption_filepaths.extend(caption_subset)
            
            # Add corresponding OVD files
            for ovd_type in self.ovd_dict.keys():
                if len(self.ovd_dict[ovd_type]) == 1:
                    ovd_path = self.ovd_dict[ovd_type][0]
                else:
                    assert len(self.ovd_dict[ovd_type]) == len(self.img_data_folders)
                    ovd_path = self.ovd_dict[ovd_type][idx]
                
                ovd_subset = [
                    os.path.join(ovd_path, os.path.basename(img_file) + '.json') 
                    for img_file in img_subset
                ]
                self.all_ovd_dict_filepaths[ovd_type].extend(ovd_subset)
        
        assert self.cumulative_lengths[-1] == TOTAL_DATASET_SIZE, "Total dataset size mismatch"
        
        # Create DataFrame for efficient slicing
        combination_dict = {
            'img_file': self.all_img_filepaths,
            'caption_file': self.all_caption_filepaths,
        }
        combination_dict.update({
            f'{ovd_type}_file': self.all_ovd_dict_filepaths[ovd_type] 
            for ovd_type in self.ovd_dict.keys()
        })
        
        # Verify all file lists have same length
        same_len = len(combination_dict['img_file'])
        for key, value in combination_dict.items():
            assert len(value) == same_len, f"Length mismatch for {key}"
        
        print(f'{self.subset_idx} | Transferring to DataFrame')
        self.dataset = pd.DataFrame(combination_dict).iloc[self.start_index:self.end_index]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """Get a single data sample.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Complete data sample with image, captions, and OVD annotations
        """
        if index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        
        config = copy.deepcopy(self.config)
        merged_dict = {}
        
        # Load file paths
        with timeit_context(f"{self.subset_idx} | load path"):
            row = self.dataset.iloc[index].copy()
            img_path = row['img_file']
            new_key = os.path.basename(img_path)
            merged_dict['new_key'] = new_key
        
        # Read image bytes
        with timeit_context(f"{self.subset_idx} | read bytes"):
            with open(img_path, 'rb') as f:
                fd = f.fileno()
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                img_bytes = f.read()
        
        # Read caption JSON
        with timeit_context(f"{self.subset_idx} | read caption json"):
            caption_filepath = os.path.join(self.config.caption_dir, new_key + '.json')
            merged_dict = read_single_json_file(
                merged_dict, caption_filepath, self.caption_key_str, 
                self.llama3_json_folders, add_prefix=None
            )
            
        # Read OVD JSONs
        with timeit_context(f"{self.subset_idx} | read ovd json"):
            for ovd_type in self.ovd_dict.keys():
                ovd_filepath = os.path.join(
                    os.path.join(self.config.ovd_dir, ovd_type), 
                    new_key + '.json'
                )
                merged_dict = read_single_json_file(
                    merged_dict, ovd_filepath, f'{ovd_type}_file', 
                    self.ovd_dict[ovd_type], add_prefix=f'{ovd_type}_'
                )
        
        # Process image
        with timeit_context(f"{self.subset_idx} | process img"):
            image_pil = process_types_of_img_bytes(img_bytes)
            image_tensor, trans_info = self._transform_image(image_pil)
            
            config.image_w, config.image_h = image_pil.size
            
            merged_dict.update({
                'image_pil': image_pil,
                'image_tensor': image_tensor,
                'trans_info': trans_info,
                'config': config
            })
        
        return merged_dict
    
    def _transform_image(self, pil_img):
        """Transform PIL image to tensor format.
        
        Args:
            pil_img (PIL.Image): Input PIL image
            
        Returns:
            tuple: (image_tensor, transform_info)
        """
        WW, HH = pil_img.size
        arr = np.array(pil_img)
        
        # Create transform info
        trans_info = {
            "performed_flip": False, 
            "performed_scale": 1.0, 
            'crop_y': 0.0, 
            'crop_x': 0.0, 
            "WW": WW, 
            'HH': HH
        }
        
        # Normalize and convert to tensor
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])
        
        return torch.tensor(arr), trans_info