import os
import json
import copy
import random
import numpy as np
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import io


def highest_frequency_element(input_list):
    """Return the most frequent element, with random selection if tied."""
    if not input_list:
        return None
    frequency_count = Counter(input_list)
    max_frequency = max(frequency_count.values())
    max_elements = [item for item, count in frequency_count.items() if count == max_frequency]
    return random.choice(max_elements)


def process_image(image_data, to_np=False):
    """Process different image formats and convert to RGB."""
    image = Image.open(io.BytesIO(image_data))
    
    if image.mode in ('RGBA', 'LA') or \
       image.mode == 'P' and 'transparency' in image.info or \
       image.mode in ('RGBa', 'La') or \
       image.mode == 'CMYK':
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')
    
    if image.mode == 'RGBA':
        img = np.array(image)
        alpha = img[:, :, 3, np.newaxis]
        img = alpha / 255 * img[..., :3] + 255 - alpha
        img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)
        image = Image.fromarray(img, 'RGB')
    
    if to_np:
        image = np.array(image)    
    return image


class OvdDataset(Dataset):
    """Dataset for loading images and their corresponding object detection annotations."""
    
    def __init__(self, box_bsz=40, max_len=392, min_len=28, secondary_max_len=224):
        super().__init__()
        
        self.box_config = {
            'box_bsz': box_bsz,
            'max_len': max_len,
            'min_len': min_len,
            'secondary_max_len': secondary_max_len,
        }

        # Load image directory mappings
        img_dirs_path = "img_dirs.json"
        if not os.path.exists(img_dirs_path):
            raise FileNotFoundError(f"Required file {img_dirs_path} not found. Please ensure your data is properly set up.")
        
        with open(img_dirs_path, 'r') as f:
            self.img_dir_dict = json.load(f)
        
        # Load todo list for processing
        todo_path = "todo_img_ovd_map.json"
        if not os.path.exists(todo_path):
            raise FileNotFoundError(f"Required file {todo_path} not found. Please ensure your data is properly set up.")
            
        with open(todo_path, 'r') as f:
            self.todo_dict = json.load(f)
            
        self.dataset_source = []
        count = 0
        
        for k, v in self.todo_dict.items():
            img_path = os.path.join(self.img_dir_dict[k], k)
            ovd_paths = list(v.values())
            self.dataset_source.append((k, img_path, ovd_paths))
            count += len(ovd_paths)
            
        print(f'Dataset created with {count} OVD samples')
        
    def __len__(self):
        return len(self.dataset_source)

    def __getitem__(self, index):
        if index >= len(self.dataset_source):
            raise IndexError 
        
        new_key, img_path, ovd_paths = self.dataset_source[index]
        
        # Load image
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            img_pil = process_image(img_bytes, to_np=False)
        except Exception as e:
            print(f'Error reading image: {img_path}')
            img_pil = None
        
        out_dict = {
            'new_key': new_key,
            'img': img_pil,
            'box_config': copy.deepcopy(self.box_config),
            'vlm_caption': None,  # Not used in this stage
            'ovd_list': []
        }
        
        # Load OVD annotations
        for ovd_path in ovd_paths:
            out_dict_single = {
                'merged_ovd_path': ovd_path
            }
            
            ovd_name = os.path.splitext(os.path.basename(ovd_path))[0]
            sub_sample = ovd_name[-1]
            assert new_key == ovd_name[:-2], f'Key mismatch: {new_key} | {ovd_name}'
            
            if img_pil is not None:
                W, H = img_pil.size
            else:
                W, H = (1024, 1024)  # Fallback size
            
            with open(ovd_path, 'r') as f:
                json_data = json.load(f)
            
            assert len(json_data['OV_merged']) == len(json_data['bboxes'])
            
            out_dict_single['ov_str'] = json_data['OV_merged']
            # Convert normalized coordinates to absolute pixel coordinates
            out_dict_single['bboxes'] = [
                [float(x0*W), float(y0*H), float(x1*W), float(y1*H)]
                for x0, y0, x1, y1 in json_data['bboxes']
            ]
            out_dict_single['sub_sample'] = sub_sample 
            
            assert json_data['new_key'] == new_key
            out_dict['ovd_list'].append(out_dict_single)
        
        return out_dict


class BoxDataset(Dataset):
    """Dataset for individual bounding box crops from images."""
    
    def __init__(self, batch_img, max_len=392, min_len=28, secondary_max_len=224):
        super().__init__()
        self.max_len = max_len
        self.min_len = min_len
        self.secondary_max_len = secondary_max_len
        
        self.new_key_list = batch_img['new_key_list']
        self.img_list = batch_img['img_list']
        self.vlm_caption_list = batch_img['vlm_caption_list']
        self.ovd_batch = batch_img['ovd_batch']
        
        # Create list of all bounding boxes with metadata
        self.box_detail_list = []
        for idx, ovd_list in enumerate(self.ovd_batch):
            for ovd in ovd_list:
                sub_sample = ovd['sub_sample']
                ovd_path = ovd['merged_ovd_path']
                
                for bdx, box in enumerate(ovd['bboxes']):
                    ov = ovd['ov_str'][bdx]
                    self.box_detail_list.append((box, idx, bdx, ov, sub_sample, ovd_path))
        
        # Sort by area (largest first) for efficient batching
        area_list = []
        for idx, bundle in enumerate(self.box_detail_list):
            box = bundle[0]
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            ratio = self.max_len / max(w, h)
            resized_w = int(w * ratio)
            resized_h = int(h * ratio)
            
            if min(resized_w, resized_h) < self.min_len:
                ratio = self.min_len / min(resized_w, resized_h)
                resized_w = max(self.min_len, int(resized_w * ratio))
                resized_h = max(self.min_len, int(resized_h * ratio))
            elif min(resized_w, resized_h) > self.secondary_max_len:
                ratio = self.secondary_max_len / min(resized_w, resized_h)
                resized_w = max(self.secondary_max_len, int(resized_w * ratio))
                resized_h = max(self.secondary_max_len, int(resized_h * ratio))
            
            area_list.append((resized_w * resized_h, idx))
            
        area_list.sort(key=lambda x: x[0], reverse=True)
        order_list = [area_list[idx][1] for idx in range(len(area_list))]
        self.box_detail_list = [self.box_detail_list[idx] for idx in order_list]
        
    def __len__(self):
        return len(self.box_detail_list)

    def __getitem__(self, index):
        box, idx, bdx, ov, sub_sample, ovd_path = self.box_detail_list[index]
        
        new_key = self.new_key_list[idx]
        img_pil = self.img_list[idx]
        vlm_caption = self.vlm_caption_list[idx]
        
        # Crop and resize the bounding box region
        x1, y1, x2, y2 = box
        cropped_region = img_pil.crop((x1, y1, x2, y2))
        cropped_img = cropped_region.copy()
        
        assert cropped_img.size[0] * cropped_img.size[1] > 0, f'Invalid crop size: {cropped_img.size} | {box}'
        
        ratio = self.max_len / max(cropped_img.size)
        resized_w = int(cropped_img.size[0] * ratio)
        resized_h = int(cropped_img.size[1] * ratio)
        
        if min(resized_w, resized_h) < self.min_len:
            ratio = self.min_len / min(resized_w, resized_h)
            resized_w = max(self.min_len, int(resized_w * ratio))
            resized_h = max(self.min_len, int(resized_h * ratio))
        elif min(resized_w, resized_h) > self.secondary_max_len:
            ratio = self.secondary_max_len / min(resized_w, resized_h)
            resized_w = max(self.secondary_max_len, int(resized_w * ratio))
            resized_h = max(self.secondary_max_len, int(resized_h * ratio))    
        
        cropped_img = cropped_img.resize((resized_w, resized_h), resample=Image.LANCZOS)
        
        return {
            'idx': idx,
            'bdx': bdx,
            'new_key': new_key,
            'merged_ovd_path': ovd_path,
            'sub_sample': sub_sample,
            'cropped_img': cropped_img,
            'ov': ov,
            'box': box,
            'original_img': None,
            'vlm_caption': vlm_caption,
            'img_size': img_pil.size,
        }


def box_collate_fn(batch):
    """Collate function for box dataset batches."""
    return {
        'bdx_list': [sample['bdx'] for sample in batch],
        'new_key_list': [sample['new_key'] for sample in batch],
        'sub_sample_list': [sample['sub_sample'] for sample in batch],
        'cropped_img_list': [sample['cropped_img'] for sample in batch],
        'ov_str': [sample['ov'] for sample in batch],
        'box_list': [sample['box'] for sample in batch],
        'merged_ovd_path_list': [sample['merged_ovd_path'] for sample in batch],
        'img_size_list': [sample['img_size'] for sample in batch],
        'original_img_list': [sample['original_img'] for sample in batch],
        'vlm_caption_list': [sample['vlm_caption'] for sample in batch],
    }


def img_collate_fn(batch):
    """Collate function that creates a box loader from image batch."""
    new_key_list = []
    img_list = []
    vlm_caption_list = []
    ovd_batch = []
    
    box_config = batch[0]['box_config']
    box_bsz = box_config['box_bsz']
    max_len = box_config['max_len']
    min_len = box_config['min_len']
    secondary_max_len = box_config['secondary_max_len']
    
    # Filter out failed image loads
    for sample in batch:
        if sample['img'] is not None:
            img_list.append(sample['img'])
            new_key_list.append(sample['new_key'])
            vlm_caption_list.append(sample['vlm_caption'])
            ovd_batch.append(sample['ovd_list'])
    
    img_batch = {
        'new_key_list': new_key_list,
        'img_list': img_list,
        'vlm_caption_list': vlm_caption_list,
        'ovd_batch': ovd_batch,
    }
    
    box_dataset = BoxDataset(
        img_batch, 
        max_len=max_len, 
        min_len=min_len,
        secondary_max_len=secondary_max_len
    )
    
    box_loader = DataLoader(
        box_dataset, 
        collate_fn=box_collate_fn,
        batch_size=box_bsz, 
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    return box_loader