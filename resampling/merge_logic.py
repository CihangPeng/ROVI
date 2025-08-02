import os
import json
import math
import random

import torch
import numpy as np

import supervision as sv
import torchvision
from torchvision.ops import nms

from ovd_utils import InstanceData

# Scoring functions for different sampling stages
def stage_3_area_distance_method(input_tuple, config, extra_score_part=None):
    """Calculate score for stage 3 sampling based on area and distance."""
    area, distance = input_tuple
    alpha = config.stage_3_basic_score
    beta = config.stage_3_area_weight
    gamma = config.stage_3_distance_weight
    return alpha + beta * area + gamma * distance


def stage_4_ovd_merge_method(input_sum, config, extra_score_part=None):
    """Calculate score for stage 4 OVD merge sampling."""
    alpha = config.stage_4_basic_score
    beta = config.stage_4_iou_weight
    gamma = config.stage_4_denominator_basic
    theta = config.stage_4_new_label_bonus
    extra_part = extra_score_part if extra_score_part is not None else 0
    return alpha + beta * input_sum / (input_sum + gamma) + theta * extra_part


def stage_5_merged_select_method(input_sum, config, extra_score_part=None):
    """Calculate score for stage 5 merged selection sampling.""" 
    alpha = config.stage_5_basic_score
    beta = config.stage_5_numerator
    gamma = config.stage_5_denominator_basic
    theta = config.stage_5_new_label_bonus
    extra_part = extra_score_part if extra_score_part is not None else 0
    return alpha + beta / (gamma + input_sum) + theta * extra_part


# Geometric calculation functions
def calc_ioa(box1, box2, iou_instead=False):
    """Calculate Intersection over Area (IoA) or IoU between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
        iou_instead: If True, calculate IoU instead of IoA
        
    Returns:
        tuple: (score_b1, score_b2) - IoA/IoU scores for each box
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    if width == 0 or height == 0:
        return 0.0, 0.0
    
    intersection_area = width * height
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    if not iou_instead:
        score_b1 = intersection_area / box1_area
        score_b2 = intersection_area / box2_area
    else:
        iou = intersection_area / (box1_area + box2_area - intersection_area)
        score_b1 = iou
        score_b2 = iou
    
    return score_b1, score_b2


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes using torchvision."""
    box1 = box1.unsqueeze(0)
    box2 = box2.unsqueeze(0)
    iou = torchvision.ops.box_iou(box1, box2)
    return iou.item()


def bbox_overlap_ratio(box1, box2):
    """Calculate overlap ratio between two bboxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    overlap_area = (x2 - x1) * (y2 - y1)
    
    return overlap_area / min(area1, area2)


def get_single_box_area(bbox):
    """Calculate normalized area of a single bounding box."""
    assert len(bbox) == 4
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    assert area <= 1.0
    return area


def merge_bboxes(box1, box2):
    """Merge two bounding boxes by taking their union."""
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return torch.tensor([x1, y1, x2, y2], dtype=box1.dtype, device=box1.device)


# Label and data manipulation functions
def merge_and_update_labels(ov_lists, label_lists):
    """Merge object vocabulary lists and update corresponding labels.
    
    Args:
        ov_lists: List of object vocabulary lists
        label_lists: List of label arrays corresponding to each vocabulary
        
    Returns:
        tuple: (merged_ov_list, rearranged_labels)
    """
    merged_set = set()
    for lst in ov_lists:
        merged_set.update(lst)

    merged_list = list(merged_set)
    merged_list.sort()

    string_to_new_index = {string: index for index, string in enumerate(merged_list)}

    rearranged_labels = []
    for i, labels in enumerate(label_lists):
        new_labels = [string_to_new_index[ov_lists[i][idx]] for idx in labels]
        rearranged_labels.append(new_labels)
        
    return merged_list, rearranged_labels


def select_lists_with_indices(bboxes, scores, labels, indices, ovd_belonging=None):
    """Select elements from lists using given indices."""
    bboxes = [bboxes[i] for i in indices]
    scores = [scores[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    if ovd_belonging is not None:
        ovd_belonging = [ovd_belonging[i] for i in indices]
    
    return bboxes, scores, labels, ovd_belonging, indices


def select_topk_score(bboxes, scores, labels, top_k=80):
    """Select top-k highest scoring detections."""
    scores_array = np.array(scores)
    bboxes_array = np.array(bboxes)
    labels_array = np.array(labels)

    top_indices = np.argsort(scores_array)[-top_k:][::-1]

    top_scores = scores_array[top_indices].tolist()
    top_bboxes = bboxes_array[top_indices].tolist()
    top_labels = labels_array[top_indices].tolist()

    return top_bboxes, top_scores, top_labels


def select_score_thr(bboxes, scores, labels, score_thr=0.15):
    """Filter detections by score threshold."""
    select_indices = [idx for idx, score in enumerate(scores) if score >= score_thr]
    return select_lists_with_indices(bboxes, scores, labels, select_indices)


# Sampling and probability functions
def uniform_choose_tmp_scores(t_scores, score_func, config, extra_score_parts=None):
    """Weighted random selection based on computed scores.
    
    Args:
        t_scores: List of input scores/tuples
        score_func: Function to compute final scores
        config: Configuration object
        extra_score_parts: Additional score components
        
    Returns:
        int: Index of chosen element, -1 if selection failed
    """
    assert isinstance(t_scores, list)
    if extra_score_parts is None:
        extra_score_parts = [None] * len(t_scores)
    
    t_scores = [
        score_func(t_score, config, extra_score_part) 
        for t_score, extra_score_part in zip(t_scores, extra_score_parts)
    ]
    
    t_sum = sum(t_scores)
    
    # Build cumulative distribution
    t_offset = [0]
    offset = 0
    for t_score in t_scores:
        offset += t_score
        t_offset.append(offset)
    
    rand_score = random.uniform(0, t_sum)
    
    # Find selected index
    for i in range(len(t_offset) - 1):
        if t_offset[i] <= rand_score < t_offset[i + 1]:
            return i
    
    return -1


# Matrix operations for overlap detection
def create_same_ov_view(labels, ovd_belonging):
    """Create matrix indicating same labels from different OVD sources."""
    n = len(labels)
    same_ov_view = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if labels[i] == labels[j] and ovd_belonging[i] != ovd_belonging[j]:
                    same_ov_view[i][j] = 1
    
    return same_ov_view


def get_ioa_matrix_ndarray(bboxes, same_ov_view=None, iou_instead=False, 
                          is_print=False, same_ov_punish=False, labels=None):
    """Compute IoA/IoU matrix for all bbox pairs."""
    N = len(bboxes)
    ioa_map = [[0] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(i + 1, N):
            if same_ov_view is not None and same_ov_view[i][j] == 0:
                continue
            
            ioa_b1, ioa_b2 = calc_ioa(bboxes[i], bboxes[j], iou_instead)
            ioa_map[i][j] = ioa_b1
            ioa_map[j][i] = ioa_b2
            
            if same_ov_punish and labels[i] == labels[j]:
                ioa_map[i][j] *= 2.0
                ioa_map[j][i] *= 2.0
    
    if is_print:
        for row in ioa_map:
            print(', '.join([f'{x:.2f}' for x in row]))
    
    return np.array(ioa_map)


def get_iou_sum_with_already_sampled(bboxes, already_sampled_boxes, labels, already_sampled_labels):
    """Calculate IoU sum with already sampled boxes for punishment."""
    N = len(bboxes)
    M = len(already_sampled_boxes)
    
    if M == 0:
        return np.array([])
    
    iou_map = [[0] * M for _ in range(N)]
    
    for i in range(N):
        for j in range(M):
            iou, _ = calc_ioa(bboxes[i], already_sampled_boxes[j], iou_instead=True)
            iou_map[i][j] = iou
            
            if labels[i] == already_sampled_labels[j]:
                iou_map[i][j] *= 2.0
    
    np_array = np.array(iou_map)
    return 2.0 / (np.sum(np_array, axis=1) + 0.5) if np_array.size > 0 else np.array([])


def sampling_from_ioa_matrix_ndarray(mat, keep_indices, config, score_func, rm_row, rm_col, 
                                   extra_score_parts=None, selected_overlap_punish=None, rm_overlap=False):
    """Sample from IoA matrix and update matrix accordingly."""
    mat_sum = np.sum(mat, axis=1)
    chosen_idx = uniform_choose_tmp_scores(mat_sum.tolist(), score_func, config, extra_score_parts)
    
    if chosen_idx == -1:
        return None, None, None
    
    mapped_chosen_idx = keep_indices[chosen_idx]
    
    if rm_overlap:
        row = mat[chosen_idx]
        related_idx = np.nonzero(row)[0]
        
        to_rm_l = []
        for idx in related_idx:
            ioa_to_check = mat[idx, chosen_idx]
            if ioa_to_check > config.stage_5_overlap_ioa_threshold:
                to_rm_l.append(idx)
                
        if chosen_idx not in to_rm_l:
            to_rm_l.append(chosen_idx)
            
        mat = np.delete(mat, to_rm_l, axis=0)
        mat = np.delete(mat, to_rm_l, axis=1)
        keep_indices = np.delete(keep_indices, to_rm_l, axis=0)
        
    else:
        if rm_row:
            mat = np.delete(mat, chosen_idx, axis=0)
            
        if rm_col:
            mat = np.delete(mat, chosen_idx, axis=1)
        
        keep_indices = np.delete(keep_indices, chosen_idx, axis=0)
        
    if selected_overlap_punish is not None:
        assert rm_col == False
        mat[:, mapped_chosen_idx] = selected_overlap_punish * mat[:, mapped_chosen_idx]
    
    return mat, keep_indices, mapped_chosen_idx


# Main sampling function
def sampling_k_from_bboxes_ioa(bboxes, scores, labels, already_sampled_boxes, already_sampled_labels,
                              sample_k, config, stage, ovd_belonging=None):
    """Main function for sampling k bboxes using IoA-based selection.
    
    Args:
        bboxes, scores, labels: Detection results
        already_sampled_boxes, already_sampled_labels: Previously sampled items
        sample_k: Number of samples to select
        config: Configuration object
        stage: Sampling stage ('stage_4', 'stage_5', 'stage_5_no_overlap')
        ovd_belonging: OVD source information
        
    Returns:
        tuple: Selected bboxes, scores, labels, ovd_belonging, and indices
    """
    if stage != 'stage_5_no_overlap' and sample_k >= len(bboxes):
        return bboxes, scores, labels, ovd_belonging, []

    assert stage in ['stage_4', 'stage_5', 'stage_5_no_overlap']
    
    # Configure sampling parameters based on stage
    if stage == 'stage_4':
        score_func = stage_4_ovd_merge_method
        rm_row, rm_col, rm_overlap = True, True, False
        assert ovd_belonging is not None
        same_ov_view = create_same_ov_view(labels, ovd_belonging)
        ioa_map = get_ioa_matrix_ndarray(bboxes, same_ov_view, iou_instead=True)
        selected_overlap_punish = None
        already_sampled_punish_weight = config.stage_4_already_sampled_punish_weight
        
    elif stage == 'stage_5':
        score_func = stage_5_merged_select_method
        rm_row, rm_col, rm_overlap = True, False, False
        same_ov_view = None
        selected_overlap_punish = config.stage_5_selected_overlap_punish
        ioa_map = get_ioa_matrix_ndarray(bboxes, same_ov_view, iou_instead=False, 
                                       same_ov_punish=True, labels=labels)
        already_sampled_punish_weight = config.stage_5_already_sampled_punish_weight
        
    elif stage == 'stage_5_no_overlap':
        score_func = stage_5_merged_select_method
        rm_row, rm_col, rm_overlap = True, True, True
        same_ov_view = None
        selected_overlap_punish = None
        ioa_map = get_ioa_matrix_ndarray(bboxes, same_ov_view, iou_instead=False, 
                                       same_ov_punish=True, labels=labels)
        already_sampled_punish_weight = config.stage_5_already_sampled_punish_weight

    # Calculate punishment for already sampled boxes
    already_sampled_punish_per_box = already_sampled_punish_weight * get_iou_sum_with_already_sampled(
        bboxes, already_sampled_boxes, labels, already_sampled_labels)
    
    assert len(already_sampled_punish_per_box) == len(bboxes)
    
    keep_indices = np.arange(len(bboxes))
    selected_indices = []
    
    while len(selected_indices) < sample_k:
        assert len(keep_indices) == len(ioa_map)
        labels_selected = [labels[selected_index] for selected_index in selected_indices]
        
        # Calculate extra score parts based on stage
        if stage == 'stage_4':
            extra_score_parts = [
                config.stage_4_new_label_bonus if labels[keep_index] not in labels_selected else 0.0 
                for keep_index in keep_indices
            ]
        else:  # stage_5 or stage_5_no_overlap
            extra_score_parts = [
                config.stage_5_new_label_bonus if labels[keep_index] not in labels_selected else 0.0 
                for keep_index in keep_indices
            ]
            # Add area bonus for stage 5
            extra_score_parts = [
                extra_score + config.stage_5_area_weight / config.stage_5_new_label_bonus * get_single_box_area(bboxes[keep_index])
                for extra_score, keep_index in zip(extra_score_parts, keep_indices)
            ]
        
        # Add already sampled punishment
        extra_score_parts = [
            extra_score + already_sampled_punish 
            for extra_score, already_sampled_punish in zip(extra_score_parts, already_sampled_punish_per_box)
        ]
        
        # Sample next box
        ioa_map, keep_indices, mapped_chosen_idx = sampling_from_ioa_matrix_ndarray(
            ioa_map, keep_indices, config, score_func=score_func,
            rm_row=rm_row, rm_col=rm_col, extra_score_parts=extra_score_parts,
            selected_overlap_punish=selected_overlap_punish, rm_overlap=rm_overlap
        )
        
        if ioa_map is None:
            break
            
        selected_indices.append(mapped_chosen_idx)
    
    return select_lists_with_indices(bboxes, scores, labels, selected_indices, ovd_belonging)


# NMS and overlap removal functions
def is_completely_inside(box1, box2, score1, score2):
    """Check if box1 is completely inside box2 with score validation."""
    is_1_in_2 = (box1[0] >= box2[0] and box1[1] >= box2[1] and
                box1[2] <= box2[2] and box1[3] <= box2[3])
    is_score_pass = (score2 >= 0.25) or (score2 >= 0.5 * score1)
    return is_1_in_2 and is_score_pass


def find_overlap_bboxs(class_bboxes, class_scores):
    """Find and remove completely overlapping bounding boxes."""
    merged_bboxes = []
    merged_scores = []
    skip_indices = set()
    to_keep_index = []
    
    for i in range(len(class_bboxes)):
        if i in skip_indices:
            continue
            
        box1 = class_bboxes[i]
        score1 = class_scores[i]
        
        for j in range(i + 1, len(class_bboxes)):
            if j in skip_indices:
                continue
                
            box2 = class_bboxes[j]
            score2 = class_scores[j]
            
            if is_completely_inside(box1, box2, score1, score2):
                box1 = None
                skip_indices.add(i)
                break
            elif is_completely_inside(box2, box1, score2, score1):
                skip_indices.add(j)
        
        if box1 is not None:
            merged_bboxes.append(box1)
            merged_scores.append(score1)
            to_keep_index.append(i)
    
    if merged_bboxes:
        merged_bboxes = torch.stack(merged_bboxes)
        merged_scores = torch.tensor(merged_scores, dtype=class_scores.dtype, device=class_scores.device)
    
    return merged_bboxes, merged_scores, to_keep_index


def label_independent_nms(bboxes, scores, labels, iou_threshold=0.5):
    """Apply NMS independently for each label class."""
    keep = []
    unique_labels = labels.unique()
    
    for label in unique_labels:
        class_indices = (labels == label).nonzero(as_tuple=True)[0]
        class_keep = nms(bboxes[class_indices], scores[class_indices], iou_threshold)
        keep.append(class_indices[class_keep])
        
    if len(keep) == 0:
        return torch.tensor([], dtype=torch.long, device=bboxes.device)
    
    return torch.cat(keep)


def label_independent_overlap_remove(bboxes, scores, labels):
    """Remove overlapping boxes independently for each label class."""
    keep = []
    unique_labels = labels.unique()
    
    for label in unique_labels:
        class_indices = (labels == label).nonzero(as_tuple=True)[0]
        _, _, to_keep_index = find_overlap_bboxs(bboxes[class_indices], scores[class_indices])
        keep.append(class_indices[to_keep_index])
        
    if len(keep) == 0:
        return torch.tensor([], dtype=torch.long, device=bboxes.device)
    
    return torch.cat(keep)


# Bbox merging functions
def recalculate_iou(bboxes):
    """Calculate IoU matrix for list of bboxes."""
    iou_map = [[0] * len(bboxes) for _ in range(len(bboxes))]
    
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            iou_map[i][j] = calculate_iou(bboxes[i], bboxes[j])
    
    return iou_map


def recalculate_overlap(bboxes):
    """Calculate overlap ratio matrix for list of bboxes."""
    iou_map = [[0] * len(bboxes) for _ in range(len(bboxes))]
    
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            iou_map[i][j] = bbox_overlap_ratio(bboxes[i], bboxes[j])
    
    return iou_map


def bbox_slice_merge(bboxes, scores, labels, merge_threshold=0.3):
    """Merge overlapping boxes for OWLv2 model results."""
    keep = []
    unique_labels = labels.unique()
    
    new_bboxes = []
    new_scores = []
    new_labels = []
    
    for label in unique_labels:
        class_indices = (labels == label).nonzero(as_tuple=True)[0]
        class_bboxes = bboxes[class_indices]
        class_scores = scores[class_indices]
        
        bboxes_map = []
        scores_map = []
        
        for i in range(len(class_bboxes)):
            bboxes_map.append(class_bboxes[i])
            scores_map.append(class_scores[i])
        
        while True:
            recalculate_map = recalculate_overlap(bboxes_map)
            
            next_merge_i = -1
            next_merge_j = -1
            max_merge_score = 0.0
            
            for i in range(len(bboxes_map)):                
                for j in range(i + 1, len(bboxes_map)):
                    if recalculate_map[i][j] > merge_threshold:                        
                        max_merge_score = recalculate_map[i][j]
                        next_merge_i = i
                        next_merge_j = j
                
            if next_merge_i == -1 or max_merge_score < merge_threshold:
                break
            
            bboxes_map.append(merge_bboxes(bboxes_map[next_merge_i], bboxes_map[next_merge_j]))
            
            if scores_map[next_merge_i] > scores_map[next_merge_j]:
                scores_map.append(scores_map[next_merge_i])
            else:
                scores_map.append(scores_map[next_merge_j])
            
            index_max = max(next_merge_i, next_merge_j)
            index_min = min(next_merge_i, next_merge_j)
            bboxes_map.pop(index_max)
            bboxes_map.pop(index_min)
            scores_map.pop(index_max)
            scores_map.pop(index_min)

        new_bboxes.extend(bboxes_map)
        new_scores.extend(scores_map)
        labels_map = [label] * len(bboxes_map)
        new_labels.extend(labels_map)
        
    new_bboxes = torch.stack(new_bboxes, dim=0)
    new_scores = torch.stack(new_scores, dim=0)
    new_labels = torch.stack(new_labels, dim=0)
        
    return InstanceData(bboxes=new_bboxes, scores=new_scores, labels=new_labels)


def bbox_OV_merge(ori_bboxes, ori_scores, ori_labels, iou_threshold=0.9):
    """Merge bounding boxes with high IoU overlap."""
    bboxes = ori_bboxes.clone()
    scores = ori_scores.clone()
    labels = ori_labels.clone()
    
    # Random shuffle for fair processing
    shuffle_indices = torch.randperm(len(bboxes))
    
    if len(bboxes.shape) == 2:
        bboxes = bboxes[shuffle_indices, :]
        scores = scores[shuffle_indices]
        labels = labels[shuffle_indices]
    
    bboxes_map = []
    scores_map = []
    labels_map = []
    
    for i in range(len(bboxes)):
        bboxes_map.append(bboxes[i])
        scores_map.append([scores[i]])
        labels_map.append([labels[i]])
        
    assert len(bboxes_map) == len(scores_map)
    
    # Iteratively merge boxes with high IoU
    while True:
        iou_map = recalculate_iou(bboxes_map)
        
        next_merge_i = -1
        next_merge_j = -1
        max_iou = 0.0
        
        for i in range(len(bboxes_map)):
            for j in range(i + 1, len(bboxes_map)):
                if iou_map[i][j] > max_iou:
                    max_iou = iou_map[i][j]
                    next_merge_i = i
                    next_merge_j = j

        if next_merge_i == -1 or max_iou < iou_threshold:
            break
        
        bboxes_map.append(merge_bboxes(bboxes_map[next_merge_i], bboxes_map[next_merge_j]))
        scores_map.append(scores_map[next_merge_i] + scores_map[next_merge_j])
        labels_map.append(labels_map[next_merge_i] + labels_map[next_merge_j])
        
        # Remove merged boxes
        index_max = max(next_merge_i, next_merge_j)
        index_min = min(next_merge_i, next_merge_j)
        bboxes_map.pop(index_max)
        bboxes_map.pop(index_min)
        scores_map.pop(index_max)
        scores_map.pop(index_min)
        labels_map.pop(index_max)
        labels_map.pop(index_min)
    
    assert len(bboxes_map) == len(scores_map) == len(labels_map)
    
    merged_bboxes = torch.stack(bboxes_map)
    return merged_bboxes, scores_map, labels_map


def get_merged_ovd(bboxes, scores, labels, config):
    """Main function to process and merge OVD results according to configuration.
    
    Args:
        bboxes, scores, labels: Detection results
        config: Configuration dictionary with processing parameters
        
    Returns:
        tuple: (merged_bboxes, merged_labels) - processed detection results
    """
    first_score_top_k = config['first_score_top_k']
    score_thr = config['score_thr']
    nms_thr = config['nms_thr']
    ov_merge_iou_threshold = config['ov_merge_iou_threshold']
    max_dets = config['max_dets']
    model_type = config['model_type']
    
    # Initial score-based filtering
    if config['do_first_score_topk']:
        bboxes, scores, labels = select_topk_score(bboxes, scores, labels, top_k=first_score_top_k)
    
    pred_instances = InstanceData(
        bboxes=torch.tensor(bboxes, dtype=torch.float32, device='cpu'),
        scores=torch.tensor(scores, dtype=torch.float32, device='cpu'),
        labels=torch.tensor(labels, dtype=torch.long, device='cpu')
    )

    # Score threshold filtering
    if config['do_score_threshold']:
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    
    # Model-specific bbox merging
    if config['do_bbox_slice_merge'] and model_type == 'owlv2':
        pred_instances = bbox_slice_merge(pred_instances.bboxes, pred_instances.scores, pred_instances.labels, merge_threshold=0.3)
    
    # Overlap removal
    if config['do_label_independent_overlap_remove']:
        keep = label_independent_overlap_remove(pred_instances.bboxes, pred_instances.scores, pred_instances.labels)
        pred_instances = pred_instances[keep]
    
    # NMS
    if config['do_label_independent_nms']:
        keep = label_independent_nms(pred_instances.bboxes, pred_instances.scores, pred_instances.labels, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep]
    
    # Max detections filtering before OV merge
    if config['do_max_dets_before_ov_merge'] and len(pred_instances.scores) > max_dets:
        keep = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[keep]
    
    # OV merge processing
    if config['do_bbox_OV_merge']:
        ori_bboxes = pred_instances.bboxes
        ori_scores = pred_instances.scores
        ori_labels = pred_instances.labels
        
        if len(ori_bboxes.shape) == 2:
            merged_bboxes, _, labels_map = bbox_OV_merge(ori_bboxes, ori_scores, ori_labels, ov_merge_iou_threshold)
        else:
            merged_bboxes = ori_bboxes
            labels_map = [[ori_label_single] for ori_label_single in ori_labels]

        # Random label selection from merged groups
        merged_labels = []
        for labels_of_single_box in labels_map:
            merged_labels.append(random.choice(labels_of_single_box).item())
        
        assert len(merged_bboxes) == len(merged_labels)
        merged_bboxes = merged_bboxes.numpy().tolist()
    else:
        merged_bboxes = pred_instances.bboxes.numpy().tolist()
        merged_labels = pred_instances.labels.numpy().tolist()
    
    return merged_bboxes, merged_labels