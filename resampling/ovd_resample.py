# ============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE FOR YOUR SETUP
# ============================================================================
ROOT_DIR = '/path/to/your/root/dir'
EXPERIMENT_NAME = 'RESAMPLE'
# ============================================================================

import os
import torch
import random
import math
import sys
import io
import time
import json
import copy
import argparse
from contextlib import contextmanager
from pprint import pprint

import supervision as sv
import numpy as np
from PIL import Image
import torchvision

from merge_logic import *
from ovd_utils import find_null_ov, InstanceData
from indexed_dataset import MultiSubsetWrapper


# Context managers and utility classes
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


class LoggingContext:
    """Context manager for capturing stdout output."""
    
    def __init__(self):
        self._original_stdout = sys.stdout
        self._log_stream = io.StringIO()

    def __enter__(self):
        sys.stdout = self._log_stream
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._log_stream.getvalue()


class DictToClass:
    """Convert dictionary to class with attribute access."""
    
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)


# Configuration and setup functions
def get_config():
    """Get default configuration for OVD resampling."""
    config = {
        'root_dir': ROOT_DIR,
        'which_key': None, 
        'no_draw': True,
        'experiment_name': EXPERIMENT_NAME,
        
        # Stage 3: Equal contribution parameters
        'stage_3_expected_low_threshold': 18,
        'stage_3_basic_score': 0.05,
        'stage_3_area_weight': math.sqrt(2) * 2.0,
        'stage_3_distance_weight': 1.0,
        
        # Stage 4: OVD merge parameters
        'stage_4_basic_score': 0.1,
        'stage_4_iou_weight': 1.2,
        'stage_4_denominator_basic': 1.0,
        'stage_4_new_label_bonus': 2.0,
        'stage_4_already_sampled_punish_weight': 0.5,
        'stage_4_single_low': 20,
        'stage_4_on_stage_3': (0.3, 0.5),
        
        # Stage 5: Final selection parameters
        'stage_5_basic_score': 0.01,
        'stage_5_numerator': 99.0,
        'stage_5_area_weight': 0.4,
        'stage_5_denominator_basic': 0.8,
        'stage_5_new_label_bonus': 1.5,
        'stage_5_selected_overlap_punish': 999,
        'stage_5_already_sampled_punish_weight': 1.0,
        'stage_5_single_low': 6,
        'stage_5_single_high': 15,
        'stage_5_inferior_low_ratio': 0.25,
        'stage_5_overlap_ioa_threshold': 0.25,
        'stage_5_on_stage_4': (0.05, 0.5),
        'stage_5_extra_max': 30,
        'stage_5_sample_k_ratio': None,
        
        # Replica and processing parameters
        'max_replica_num': 5,
        'stage45_min_remain_ratio': 0.3,
        
        # Image processing parameters
        'image_size': None,
        'image_w': None,
        'image_h': None,
        'is_train': True,
        'random_crop': False,
        'random_flip': False,
        'use_min_len_instead_of_area': True,
        'min_box_len': 0.05,
        'max_box_size': 0.60,
        'min_edge_cut_area_ratio': 0.35,
        'style_ov_shared_size': 0.35,
        
        # Score thresholds per OVD type
        'yw_score_thr': 0.145,
        'ow_score_thr': 0.175,
        'gd_score_thr': 0.195,
        'od_score_thr': 0.285,
        'nms_thr': 0.4,
        
        # Fixed seed for reproducibility
        'fixed_seed': None,
    }
    return DictToClass(config)


# Image processing and geometric functions
def area_min_max_filter(bboxes, scores, labels, trans_info, config, return_tensor=True):
    """Filter bboxes by area constraints and transform coordinates.
    
    Args:
        bboxes, scores, labels: Detection results
        trans_info: Image transformation information
        config: Configuration object
        return_tensor: Whether to return tensors or lists
        
    Returns:
        tuple: Filtered results and above_max_label_set
    """
    valid_bboxes = []
    valid_scores = []
    valid_labels = []
    valid_areas = []
    
    if config.image_size is not None:
        scale_t_list = [config.image_size] * 4
    else:
        scale_t_list = [config.image_w, config.image_h, config.image_w, config.image_h]
    
    scale_t = torch.tensor(scale_t_list)
    above_max_label_set = set()
    
    for idx, bbox in enumerate(bboxes):
        x0, y0, x1, y1 = bbox
        valid, above_max, (x0, y0, x1, y1), area = \
            recalculate_box_and_verify_if_valid(x0, y0, x1, y1, trans_info, config)

        if valid:
            if return_tensor:
                valid_bboxes.append(torch.tensor([x0, y0, x1, y1]) / scale_t)
            else:
                valid_bboxes.append([x0/scale_t_list[0], y0/scale_t_list[1], x1/scale_t_list[2], y1/scale_t_list[3]])
                                     
            valid_scores.append(scores[idx])
            valid_labels.append(labels[idx])
            valid_areas.append(area)
        
        if above_max:
            above_max_label_set.add(labels[idx])
            
    return valid_bboxes, valid_scores, valid_labels, above_max_label_set, valid_areas


def get_areas_from_bboxes(bboxes):
    """Calculate areas from normalized bounding boxes."""
    areas = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        areas.append((x1-x0) * (y1-y0))
    return np.array(areas)


def get_distances_from_bboxes(bboxes):
    """Calculate distances from bbox centers to image center."""
    distances = []
    for bbox in bboxes:
        assert max(bbox) <= 1.0, "Bboxes must be normalized within [0,1]"
        x0, y0, x1, y1 = bbox
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        distances.append(math.sqrt(2) * 0.5 - math.sqrt((center_x-0.5)**2 + (center_y-0.5)**2))
    return np.array(distances)


def to_valid(x0, y0, x1, y1, config):
    """Validate and clamp bounding box coordinates."""
    max_w = config.image_size if config.image_size is not None else config.image_w
    max_h = config.image_size if config.image_size is not None else config.image_h

    valid = True
    above_max = False
    area = 0
    
    # Check if box is completely outside image
    if x0 > max_w or y0 > max_h or x1 < 0 or y1 < 0:
        valid = False
        return valid, above_max, (None, None, None, None), area
    
    ori_area = (x1 - x0) * (y1 - y0)
    
    # Clamp coordinates to image boundaries
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, max_w)
    y1 = min(y1, max_h)
    
    area = (x1 - x0) * (y1 - y0)
    
    # Check if too much area was cut off
    if area < ori_area * config.min_edge_cut_area_ratio:
        valid = False
        return valid, above_max, (None, None, None, None), area
    
    # Apply size constraints
    if config.use_min_len_instead_of_area:
        if min((x1-x0)/max_w, (y1-y0)/max_h) < config.min_box_len:
            valid = False
            return valid, above_max, (None, None, None, None), area
    else:
        if area / (max_w * max_h) < config.min_box_size:
            valid = False
            return valid, above_max, (None, None, None, None), area
        
    if area / (max_w * max_h) > config.max_box_size:
        above_max = True
        valid = False
        return valid, above_max, (None, None, None, None), area
     
    return valid, above_max, (x0, y0, x1, y1), area


def recalculate_box_and_verify_if_valid(x0, y0, x1, y1, trans_info, config):
    """Recalculate box coordinates after transformations and validate."""
    # Apply scaling and cropping transformations
    x0 = x0 * trans_info["performed_scale"] - trans_info['crop_x'] 
    y0 = y0 * trans_info["performed_scale"] - trans_info['crop_y'] 
    x1 = x1 * trans_info["performed_scale"] - trans_info['crop_x'] 
    y1 = y1 * trans_info["performed_scale"] - trans_info['crop_y'] 

    # Validate and clamp coordinates
    valid, above_max, (x0, y0, x1, y1), area = to_valid(x0, y0, x1, y1, config)

    # Apply flip transformation if needed
    if valid and trans_info["performed_flip"]:
        x0, x1 = config.image_size - x1, config.image_size - x0

    return valid, above_max, (x0, y0, x1, y1), area


# Annotation and visualization
# Adapted from supervision library with position modifications
from supervision.geometry.core import Position

class ModifiedLabelAnnotator(sv.LabelAnnotator):
    """Modified label annotator with improved text positioning."""
    
    @staticmethod
    def resolve_text_background_xyxy(center_coordinates, text_wh, position, image_size):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        image_width, image_height = image_size

        # Calculate initial position based on anchor
        if position == Position.TOP_LEFT:
            xyxy = (center_x, center_y, center_x + text_w, center_y + text_h)
        elif position == Position.TOP_RIGHT:
            xyxy = (center_x - text_w, center_y, center_x, center_y + text_h)
        elif position == Position.BOTTOM_LEFT:
            xyxy = (center_x, center_y - text_h, center_x + text_w, center_y)
        elif position == Position.BOTTOM_RIGHT:
            xyxy = (center_x - text_w, center_y - text_h, center_x, center_y)
        else:
            xyxy = (center_x, center_y, center_x + text_w, center_y + text_h)
            
        x1, y1, x2, y2 = xyxy

        # Ensure the bounding box is within the image area
        if x1 < 0:
            x1 = 0
            x2 = x1 + text_w
        if y1 < 0:
            y1 = 0
            y2 = y1 + text_h
        if x2 > image_width:
            x2 = image_width
            x1 = x2 - text_w
        if y2 > image_height:
            y2 = image_height
            y1 = y2 - text_h

        # Final clamping
        x1 = max(0, x1)
        y1 = max(0, y1)

        return x1, y1, x2, y2


# Global annotators
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=3)
MODIFIED_LABEL_ANNOTATOR = ModifiedLabelAnnotator(text_padding=4, text_scale=1.0, text_thickness=2)


# Global statistics
count_unique_labels = 0
count_bboxes = 0
ovd_stats = {}
sum_end_of_stages = [0] * 5


# Main sampling pipeline
def ovd_merge_sample(merged_dict, ovd_dict, config):
    """Main OVD sampling pipeline implementing multi-stage selection strategy.
    
    Args:
        merged_dict: Dictionary containing image and detection data
        ovd_dict: Dictionary of OVD types to process
        config: Configuration object
        
    Returns:
        list: List of sampling results (answer_list)
    """
    image_pil = merged_dict['image_pil']
    image_tensor = merged_dict['image_tensor']
    trans_info = merged_dict['trans_info']
    
    # Stage 0: Merge OV lists and rearrange labels
    all_ov_list = [merged_dict[f'{ovd_type}_OV_to_detect'] for ovd_type in ovd_dict.keys()]
    all_labels = [merged_dict[f'{ovd_type}_labels'] for ovd_type in ovd_dict.keys()]
    
    merged_ov_list, rearranged_labels = merge_and_update_labels(all_ov_list, all_labels)
    
    for i, ovd_type in enumerate(ovd_dict.keys()):
        merged_dict[f'{ovd_type}_OV_to_detect'] = merged_ov_list
        merged_dict[f'{ovd_type}_labels'] = rearranged_labels[i]
    
    # Process each OVD type through filtering stages
    ovd_out = {}
    above_max_label_set = set()
    
    for ovd_type in ovd_dict.keys():
        bboxes = merged_dict[f'{ovd_type}_bboxes']
        scores = merged_dict[f'{ovd_type}_scores']
        labels = merged_dict[f'{ovd_type}_labels']  
        OV_to_detect = merged_dict[f'{ovd_type}_OV_to_detect']
        
        assert len(bboxes) == len(scores) == len(labels), 'Length mismatch in bboxes, scores, labels'
        
        print(f'Step 0: Remove null OV - {ovd_type}')
        print(f'Before: {len(bboxes)}')
        
        # Remove null/empty object categories
        bboxes, scores, labels = find_null_ov(bboxes, scores, labels, OV_to_detect)
        print(f'After: {len(bboxes)}\n')
        
        print(f'Step 0: Score threshold - {ovd_type}')
        print(f'Before: {len(bboxes)}')
        
        # Apply score threshold
        ovd_score_thr = getattr(config, f'{ovd_type}_score_thr')
        bboxes, scores, labels, _, _ = select_score_thr(bboxes, scores, labels, ovd_score_thr)
        print(f'After: {len(bboxes)}\n')
        
        print(f'Step 1: Filter area/size & transform - {ovd_type}')
        print(f'Before: {len(bboxes)}')
        
        # Filter by area constraints and transform coordinates
        bboxes, scores, labels, above_max_labels, areas = area_min_max_filter(
            bboxes, scores, labels, trans_info, config)
        print(f'After: {len(bboxes)}\n')
        
        above_max_label_set.update(above_max_labels)
        ovd_out[ovd_type] = (bboxes, scores, labels)
        ovd_out[f'{ovd_type}_areas'] = areas
    
    # Remove shared style objects (large area objects)
    style_ov_list = [merged_ov_list[i] for i in above_max_label_set]
    
    for ovd_type in ovd_dict.keys():
        bboxes, scores, labels = ovd_out[ovd_type]
        areas = ovd_out[f'{ovd_type}_areas']
        
        print(f'Step 1: Remove shared style OV - {ovd_type}')
        print(f'Before: {len(bboxes)}')
        
        discarded_index = []
        for i in range(len(labels)):
            if (labels[i] in above_max_label_set and 
                areas[i] / (config.image_w * config.image_h) > config.style_ov_shared_size):
                discarded_index.append(i)
        
        bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in discarded_index]
        scores = [scores[i] for i in range(len(scores)) if i not in discarded_index]
        labels = [labels[i] for i in range(len(labels)) if i not in discarded_index]
        
        print(f'After: {len(bboxes)}\n')
        
        ovd_out[ovd_type] = (bboxes, scores, labels)
        ovd_out.pop(f'{ovd_type}_areas')
    
    # Stage 2: Overlap removal and NMS for each OVD type
    for ovd_type in ovd_dict.keys():
        bboxes, scores, labels = ovd_out[ovd_type]
        
        # Convert to InstanceData for processing
        if len(bboxes):
            bboxes = torch.stack(bboxes, dim=0)
        else:
            bboxes = torch.Tensor()
        
        pred_instances = InstanceData(
            bboxes=bboxes,
            scores=torch.tensor(scores, dtype=torch.float32, device='cpu'),
            labels=torch.tensor(labels, dtype=torch.long, device='cpu')
        )
        
        print(f'Step 2: Overlap removal - {ovd_type}')
        print(f'Before: {len(pred_instances.bboxes)}')
        
        # Remove overlapping boxes
        keep = label_independent_overlap_remove(
            pred_instances.bboxes, pred_instances.scores, pred_instances.labels)
        pred_instances = pred_instances[keep]
        print(f'After: {len(pred_instances.bboxes)}\n')
        
        print(f'Step 2: NMS - {ovd_type}')
        print(f'Before: {len(pred_instances.bboxes)}')
        
        # Apply non-maximum suppression
        keep = label_independent_nms(
            pred_instances.bboxes, pred_instances.scores, pred_instances.labels,
            iou_threshold=config.nms_thr)
        pred_instances = pred_instances[keep]
        print(f'After: {len(pred_instances.bboxes)}\n')
        
        ovd_out[ovd_type] = pred_instances
    
    # Stage 3: Equal contribution sampling
    len_list = [len(pred_instances.bboxes) for pred_instances in ovd_out.values()]
    max_num = max(len_list)
    min_num = min(len_list)
    
    if max_num >= config.stage_3_expected_low_threshold:
        min_num = max(min_num, config.stage_3_expected_low_threshold)
    else:
        min_num = max_num
    
    if config.fixed_seed is not None:
        random.seed(config.fixed_seed)
    rand_num = random.randint(min_num, max_num)
    
    for ovd_type in ovd_dict.keys():
        pred_instances = ovd_out[ovd_type]
        
        print(f'Step 3: Equal contribution - {ovd_type}')
        print(f'Before: {len(pred_instances.bboxes)}')
        
        if len(pred_instances.bboxes) > rand_num:
            bboxes = pred_instances.bboxes.cpu().numpy().tolist()
            areas = get_areas_from_bboxes(bboxes)
            distances = get_distances_from_bboxes(bboxes)
            
            selected_indices = []
            keep_indices = np.arange(len(bboxes))
            
            while len(selected_indices) < rand_num:
                chosen_idx = uniform_choose_tmp_scores(
                    [(area, distance) for area, distance in zip(areas, distances)], 
                    stage_3_area_distance_method, 
                    config,
                    extra_score_parts=None)
                
                mapped_chosen_idx = keep_indices[chosen_idx]
                selected_indices.append(mapped_chosen_idx)
                
                keep_indices = np.delete(keep_indices, chosen_idx, axis=0)
                areas = np.delete(areas, chosen_idx, axis=0)
                distances = np.delete(distances, chosen_idx, axis=0)
            
            keep = torch.tensor(selected_indices, device='cpu').long()
            pred_instances = pred_instances[keep]
        
        print(f'After: {len(pred_instances.bboxes)}\n')
        ovd_out[ovd_type] = pred_instances
    
    # Update global statistics
    for ovd_type in ovd_dict:
        ovd_stats[ovd_type] += len(ovd_out[ovd_type].bboxes)
    
    # Prepare for stages 4 and 5: Merge all OVD results
    pre_merged_bboxes = []
    pre_merged_scores = []
    pre_merged_labels = []
    pre_merged_ovd_type = []
    
    for ovd_type in ovd_dict.keys():
        pred_instances = ovd_out[ovd_type]
        ovd_out[ovd_type] = pred_instances.cpu().numpy()
        pre_merged_bboxes.extend(pred_instances.bboxes.tolist())
        pre_merged_scores.extend(pred_instances.scores.tolist())
        pre_merged_labels.extend(pred_instances.labels.tolist())
        pre_merged_ovd_type.extend([ovd_type] * len(pred_instances.bboxes.tolist()))

    # Skip stages 4 and 5 if only one OVD type
    if len(ovd_dict) == 1:
        print('Stages 4-5 skipped: Only one OVD type\n')
    
    # Stages 4 and 5: Multi-replica sampling with overlap punishment
    answer_list = []
    bboxes_store = copy.deepcopy(pre_merged_bboxes)
    scores_store = copy.deepcopy(pre_merged_scores)
    labels_store = copy.deepcopy(pre_merged_labels)
    ovd_belonging_store = copy.deepcopy(pre_merged_ovd_type)
    
    global count_unique_labels, count_bboxes
    count_unique_labels += len(set(pre_merged_labels))
    count_bboxes += len(pre_merged_bboxes)
    
    num_replica = 0
    sum_boxes = len(bboxes_store)
    already_sampled_boxes = []
    already_sampled_labels = []
    
    store_stage_4_weight = config.stage_4_already_sampled_punish_weight
    store_stage_5_weight = config.stage_5_already_sampled_punish_weight
    
    # Multi-replica sampling loop
    while True:
        # Update punishment weights based on replica number
        config.stage_4_already_sampled_punish_weight = num_replica * store_stage_4_weight
        config.stage_5_already_sampled_punish_weight = num_replica * store_stage_5_weight
        
        # Check termination conditions
        if num_replica >= config.max_replica_num:
            break
        
        if (len(bboxes_store) < config.stage45_min_remain_ratio * sum_boxes or
            len(bboxes_store) < min(0.5 * sum_boxes, config.stage_5_single_low)):
            break
        
        num_replica += 1
        
        # Stage 4: OVD merge sampling
        stage_4_sample_k = int(random.uniform(
            config.stage_4_on_stage_3[0],
            config.stage_4_on_stage_3[1]) * len(bboxes_store))
        
        stage_4_sample_k = max(stage_4_sample_k, config.stage_4_single_low)
        stage_4_sample_k = min(stage_4_sample_k, len(bboxes_store))
        
        num_boxes = len(bboxes_store)
        bboxes = bboxes_store
        scores = scores_store
        labels = labels_store
        ovd_belonging = ovd_belonging_store
        
        print('Step 4: OVD merge')
        if len(bboxes) > stage_4_sample_k:
            bboxes, scores, labels, ovd_belonging, selected_l0 = \
                sampling_k_from_bboxes_ioa(
                    bboxes, scores, labels,
                    already_sampled_boxes,
                    already_sampled_labels,
                    sample_k=stage_4_sample_k, 
                    config=config, 
                    stage='stage_4',
                    ovd_belonging=ovd_belonging)
        else:
            selected_l0 = list(range(num_boxes))
        
        print(f'After: {len(bboxes)}')
        
        # Print OVD contribution statistics
        ovd_belonging_dict = {ovd_type: 0 for ovd_type in ovd_dict.keys()}
        for ovd_type in ovd_belonging:
            ovd_belonging_dict[ovd_type] += 1
        print(f'OVD contribution: {ovd_belonging_dict}\n')
    
        # Stage 5: Final selection with overlap punishment
        print('Step 5: Overlap punishment')
        print(f'Before: {len(bboxes)}')

        stage_5_sample_k_low = min(
            int(config.stage_5_inferior_low_ratio * len(bboxes)), 
            config.stage_5_single_low)
        stage_5_sample_k = random.randint(stage_5_sample_k_low, config.stage_5_single_high)
        stage_5_sample_k = min(stage_5_sample_k, len(bboxes))
        
        # Ensure at least 2 samples if possible
        if len(bboxes) >= 2:
            stage_5_sample_k = max(2, stage_5_sample_k)
        
        # Apply extra maximum limit
        if config.stage_5_extra_max is not None:
            stage_5_sample_k = min(stage_5_sample_k, config.stage_5_extra_max)
        
        assert len(bboxes) >= stage_5_sample_k
        
        bboxes, scores, labels, ovd_belonging, selected_l1 = \
            sampling_k_from_bboxes_ioa(
                bboxes, scores, labels,
                already_sampled_boxes,
                already_sampled_labels,
                sample_k=stage_5_sample_k, 
                config=config, 
                stage='stage_5_no_overlap',
                ovd_belonging=ovd_belonging)
        
        # Update already sampled lists
        already_sampled_boxes.extend(bboxes)
        already_sampled_labels.extend(labels)
        
        print(f'After: {len(bboxes)}')
        
        ovd_belonging_dict = {ovd_type: 0 for ovd_type in ovd_dict.keys()}
        for ovd_type in ovd_belonging:
            ovd_belonging_dict[ovd_type] += 1
        print(f'OVD contribution: {ovd_belonging_dict}\n')
        
        # Store current result
        answer_list.append((bboxes, scores, labels, image_tensor, merged_ov_list, style_ov_list, ovd_belonging))
        
        # Update remaining boxes for next iteration
        assert num_boxes == len(bboxes_store)
        idx_arange = list(range(num_boxes))
        idx_selected = [selected_l0[x_idx] for x_idx in selected_l1]
        idx_remain = [x_idx for x_idx in idx_arange if x_idx not in idx_selected]
        
        bboxes_store, scores_store, labels_store, ovd_belonging_store, _ = \
            select_lists_with_indices(bboxes_store, scores_store, labels_store, idx_remain, ovd_belonging_store)
            
    return answer_list


def draw_annotation(answer_list, no_draw=False):
    """Create annotated images and metadata from sampling results.
    
    Args:
        answer_list: List of sampling results
        no_draw: Whether to skip image drawing
        
    Returns:
        list: List of annotation results
    """
    out_list = []
    
    for answer in answer_list:
        bboxes, scores, labels, image_tensor, merged_ov_list, style_ov_list, ovd_belonging = answer
        part_ov = [merged_ov_list[label] for label in labels]
        part_idx = [f'{i+1}' for i in range(len(labels))]
        
        corresponding_ov_str = [f'{idx} | {ov}' for idx, ov in zip(part_idx, part_ov)]
        idx_str = [f'{idx}' for idx in part_idx]
        
        assert len(image_tensor) == 3
        TC, TH, TW = image_tensor.shape
        
        # Create output dictionary
        dict_to_write = {
            'image_size': [TW, TH],
            'model_type': 'merged',
            'annotation_type': 'all_boxes',
            'OV_merged': part_ov,
            'bboxes': bboxes,
            'scores': scores,
            'ovd_belonging': ovd_belonging,
        }
        
        selected_ov_set = set()
        for label in labels:
            selected_ov_set.add(merged_ov_list[label])
        
        if not no_draw:
            image_pil_restore = torchvision.transforms.functional.to_pil_image(image_tensor * 0.5 + 0.5)
            W, H = image_pil_restore.size
            
            assert (TH == H) and (TW == W), f'Size mismatch: {TH} {H} | {TW} {W}'
            
            bboxes_scaled = [[float(x0*W), float(y0*H), float(x1*W), float(y1*H)] for x0, y0, x1, y1 in bboxes]
            
            if len(bboxes_scaled) == 0:
                out_list.append((image_pil_restore, image_pil_restore, ovd_belonging, merged_ov_list, set(), corresponding_ov_str, dict_to_write))
                continue
            
            detections = sv.Detections(
                xyxy=np.array(bboxes_scaled),
                class_id=np.array(labels),
                confidence=np.array(scores),
            )
            
            image_np = np.array(image_pil_restore)
            image_np = BOUNDING_BOX_ANNOTATOR.annotate(image_np, detections)
            image_np = MODIFIED_LABEL_ANNOTATOR.annotate(
                image_np, detections, labels=corresponding_ov_str, image_size=[TW, TH])
    
            out_list.append((Image.fromarray(image_np), image_pil_restore, ovd_belonging, merged_ov_list, selected_ov_set, corresponding_ov_str, dict_to_write))
        else:
            out_list.append((None, None, ovd_belonging, merged_ov_list, selected_ov_set, corresponding_ov_str, dict_to_write))
    
    return out_list


def main(config_ori, step_5_ratios=None, seeds=None):
    """Main function for OVD resampling pipeline.
    
    Args:
        config_ori: Original configuration object
        step_5_ratios: Optional ratios for stage 5 sampling
        seeds: Optional random seeds for reproducibility
    """
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--divide_num", type=int, required=True)
    argparser.add_argument("--data_slice_index", type=int, required=True)
    argparser.add_argument("--test_cutdown", type=int, default=9999999)
    
    args = argparser.parse_args()
    
    divide_num = args.divide_num
    data_slice_index = args.data_slice_index
    test_cutdown = args.test_cutdown
    
    # Calculate data slice boundaries
    total_num = 1012074
    divide = total_num // divide_num
    start_end = {}

    for li in range(divide_num):
        if li != divide_num - 1:
            start_end[str(li)] = [li * divide, (li + 1) * divide]
        else:
            start_end[str(li)] = [li * divide, total_num]

    start_index, end_index = start_end[str(data_slice_index)]
    end_index = min(end_index, start_index + test_cutdown)
    sample_num = end_index - start_index
    
    # Initialize dataset
    dataset = MultiSubsetWrapper(
        config_ori,
        data_slice_index,
        start_index=start_index, 
        end_index=end_index,
        cut_down=test_cutdown,
    )
    
    ovd_dict = dataset.ovd_dict
    
    # Initialize OVD statistics
    for ovd_type in ovd_dict.keys():
        ovd_stats[ovd_type] = 0
    
    assert len(dataset) == sample_num
    
    # Setup output directories
    root_dir = config_ori.root_dir
    os.makedirs(root_dir, exist_ok=True)
    
    output = os.path.join(root_dir, f'{data_slice_index}_{sample_num}_{start_index}_{end_index}')
    os.makedirs(output, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(output, 'caption'), exist_ok=True)
    os.makedirs(os.path.join(output, 'ovd'), exist_ok=True)
    for ovd_type in ovd_dict.keys():
        os.makedirs(os.path.join(output, 'ovd', ovd_type), exist_ok=True)
    
    dataset.config.caption_dir = os.path.join(output, 'caption')
    dataset.config.ovd_dir = os.path.join(output, 'ovd')
    
    output = os.path.join(output, 'ovd_merged')
    os.makedirs(output, exist_ok=True)
    
    # Check if processing is already complete
    with open("/mnt/data1/pch/copy_paste_caption_ovd/slice_ovd_merged_not_found.json", 'r') as f:
        slice_ovd_merged_not_found = json.load(f)
    
    if slice_ovd_merged_not_found[output] == 0:
        print(f'Already done | {output}')
        exit()
    else:
        os.rename(output, os.path.join(os.path.dirname(output), 'ovd_merged_incomplete_t3'))
        os.makedirs(output, exist_ok=False)
    
    # Initialize tracking variables
    ovd_box_chosen_num = {ovd_type: 0 for ovd_type in ovd_dict.keys()}
    
    # Process each sample
    for idx, sample in enumerate(dataset):
        assert sample.get('config') is not None
        config = sample['config']
        
        which_key = config.which_key
        no_draw = config.no_draw
        
        # Handle step 5 ratios and seeds
        if step_5_ratios is not None:
            config.stage_5_sample_k_ratio = step_5_ratios[idx]
            
        if seeds is not None:
            config.fixed_seed = seeds[idx]
        
        # Process sample through OVD merge pipeline
        with timeit_context(f'ovd_merge_process | {idx}'), LoggingContext() as log:
            bundle = ovd_merge_sample(sample, ovd_dict, config)
            answer_list = draw_annotation(bundle, no_draw=no_draw)
        
        # Save results for each sub-sample
        for t_idx, answer in enumerate(answer_list):
            image, image_pil, ovd_belonging, merged_ov_list, selected_ov_set, corresponding_ov_str, dict_to_write = answer
        
            new_key = sample['new_key']
            
            dict_to_write['new_key'] = new_key
            dict_to_write['sub_sample'] = t_idx
            dict_to_write['ovd_source'] = list(ovd_dict.keys())
            
            # Save JSON metadata
            json_dir = output if no_draw else os.path.join(output, 'json')
            os.makedirs(json_dir, exist_ok=True)
            
            with open(os.path.join(json_dir, f'{new_key}_{t_idx}.json'), 'w') as f:
                json.dump(dict_to_write, f, indent=4)
            
            # Update statistics
            ovd_belonging_dict = {ovd_type: 0 for ovd_type in ovd_dict.keys()}
            for ovd_type in ovd_belonging:
                ovd_belonging_dict[ovd_type] += 1
                    
            for ovd_type in ovd_dict.keys():
                ovd_box_chosen_num[ovd_type] += ovd_belonging_dict[ovd_type]
            
            # Save visualizations and logs if not no_draw
            if not no_draw:
                vlm_caption = sample['vlm_caption']
                vlm_caption = vlm_caption.replace("This is ", "", 1)
                vlm_caption = vlm_caption[0].upper() + vlm_caption[1:]
                web_caption = sample['web_caption']
                
                logged_output = log.output
                logged_output = '\n'.join(corresponding_ov_str) + \
                    '\n\n[WEB]\n' + web_caption + \
                    '\n\n[VLM]\n' + vlm_caption + \
                    '\n\n[ALL_OV]\n' + '\n'.join(merged_ov_list) + \
                    '\n\n[SELECTED_OV]\n' + '\n'.join(selected_ov_set) + \
                    '\n\n[PROCESS]\n' + logged_output
                
                config_str = '\n'.join([f'{key}: {value}' for key, value in config.__dict__.items()])
                logged_output += f'\n\n{config_str}'

                # Save annotated image
                img_dir = os.path.join(output, 'img_anno')
                os.makedirs(img_dir, exist_ok=True)
                image.save(os.path.join(img_dir, f'{new_key}_{t_idx}.jpg'), format='JPEG', quality=75)
                
                # Save process log
                txt_dir = os.path.join(output, 'txt')
                os.makedirs(txt_dir, exist_ok=True)
                with open(os.path.join(txt_dir, f'{new_key}_{t_idx}.txt'), 'w') as f:
                    f.write(logged_output)
        
        # Break if processing specific key
        if which_key is not None and sample['new_key'] == which_key:
            break
    
    # Write completion marker
    with open(os.path.join(root_dir, f'{data_slice_index}_fin.txt'), 'w') as f:
        f.write('fin')


# Utility functions for random generation (for debugging/testing)
def create_random_ratio(config, need_num=1000):
    """Create random ratios for stage 5 sampling."""
    stage_5_sample_k_ratios = []
    
    for _ in range(need_num):
        stage_5_sample_k_ratios.append(
            random.uniform(
                config.stage_5_on_stage_4[0],
                config.stage_5_on_stage_4[1]
            )
        )
    
    to_str = '\n'.join([str(r) for r in stage_5_sample_k_ratios])
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_dir, f'step_5_sample_ratio_{need_num}.txt'), 'w') as f:
        f.write(to_str)


def create_seed(need_num=1000):
    """Create random seeds for reproducibility."""
    seed_list = []
    for i in range(need_num):
        seed_list.append(random.randint(0, 9999999999))
    
    to_str = '\n'.join([str(r) for r in seed_list])
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_dir, f'seed_{need_num}.txt'), 'w') as f:
        f.write(to_str)


def read_seed(need_num=1000):
    """Read seeds from file."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    seeds = []
    with open(os.path.join(cur_dir, f'seed_{need_num}.txt'), 'r') as f: 
        for line in f:
            seeds.append(int(line.strip()))
    return seeds


def read_ratio_from_txt(need_num=1000):
    """Read ratios from file."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ratios = []
    with open(os.path.join(cur_dir, f'step_5_sample_ratio_{need_num}.txt'), 'r') as f: 
        for line in f:
            ratios.append(float(line.strip()))
    return ratios


if __name__ == '__main__':
    config = get_config()
    main(config)
    print('Done')