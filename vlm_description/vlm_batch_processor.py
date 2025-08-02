import os
import argparse
import json
import io
import time
import traceback
from contextlib import contextmanager
from datetime import datetime

import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import datasets
import numpy as np

from indexed_dataset import IndexedDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@contextmanager
def timeit_context(name="Task"):
    """Timer context manager for performance monitoring"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(f"{name} took {elapsed_time:.6f} seconds")


def build_transform(input_size):
    """Build image preprocessing transform"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Dynamically preprocess image into multiple tiles"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target aspect ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) 
        for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split image into tiles
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    # Add thumbnail if requested
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def load_image(image, input_size=448, max_num=6):
    """Load and preprocess single image"""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)


def process_types_of_img(image_data):
    """Handle different image formats and transparency"""
    image = Image.open(io.BytesIO(image_data))
    
    if (image.mode in ('RGBA', 'LA') or 
        image.mode == 'P' and 'transparency' in image.info or
        image.mode in ('RGBa', 'La') or 
        image.mode == 'CMYK'):
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')
    
    # Handle transparency by blending with white background
    if image.mode == 'RGBA':
        img = np.array(image)
        alpha = img[:, :, 3, np.newaxis]
        img = alpha / 255 * img[..., :3] + 255 - alpha
        img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)
        image = Image.fromarray(img, 'RGB')
        
    return image


def process_image(img_bytes, max_num=6):
    """Process image bytes into PIL and tensor format"""
    img_pil = process_types_of_img(img_bytes)
    pixel_values = load_image(img_pil, max_num=max_num)
    return img_pil, pixel_values


def collate_fn(batch, max_num=6):
    """DataLoader collate function for batch processing"""
    new_key_list = []
    caption_list = []
    png_list = []
    img_processed_list = []
    
    for sample in batch:
        new_key_list.append(sample['new_key'])
        caption_list.append(sample['caption'])
        
        img_bytes = sample['png']
        img_pil, pixel_values_sample = process_image(img_bytes, max_num=max_num)
        
        png_list.append(img_pil)
        img_processed_list.append(pixel_values_sample)
    
    image_counts = [pv.size(0) for pv in img_processed_list]
    pixel_values = torch.cat(img_processed_list, dim=0)
    
    return {
        'new_key': new_key_list,
        'pixel_values': pixel_values, 
        'image_counts': image_counts,
        'png': png_list, 
        'caption': caption_list,
    }


def write_timestamp_to_file(file_path):
    """Write completion timestamp to file"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a+') as file:
        file.write(current_time + '\n')
        file.seek(0)
        print(file.read())


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="VLM Batch Image Captioning")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--root_dir", type=str, required=True, help="Root output directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--kill_index", type=int, required=True, help="Kill signal index")
    parser.add_argument("--bsz", type=int, default=4, help="Batch size")
    parser.add_argument("--max_num", type=int, default=12, help="Maximum number of image tiles")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature")
    parser.add_argument("--data_slice_index", type=int, required=True, help="Data slice index for distributed processing")
    parser.add_argument("--gpu_num", type=int, required=True, help="Total number of GPUs")
    parser.add_argument("--test_cutdown", type=int, default=9999999, help="Limit number of samples for testing")
    
    args = parser.parse_args()
    
    # Get current script directory for relative paths
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Setup directories
    os.makedirs(args.root_dir, exist_ok=True)
    img_dir = os.path.join(args.root_dir, "images")
    caption_dir = os.path.join(args.root_dir, "captions")
    query_dir = os.path.join(args.root_dir, "queries")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(caption_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    
    # Setup generation prompt
    query = ("Describe the image in one paragraph, in prompt style. "
             "Put important things first. "
             "First introduce the overall picture style, such as 'photograph, cartoon, 3d render'. "
             "Focus on objects, but be precise. "
             "You shall specify all the clear objects within the image. "
             "Omit unclear or tiny objects that are not major. "
             "Pay attention to people, characters, and human activities. "
             "Be brief but complete. "
             "DO NOT mention texts. "
             "DO NOT include atmosphere, impression, or feeling. "
             "DO NOT use guessing words like 'suggest, appears, indicates, likely, might'; use firm statements instead. "
             "DO NOT include anything you are not sure of. "
             "DO NOT use negative statements such as 'with no; There are no'. "
             "If the lighting is special, put it at the end, but ignore the normal situation. "
             "Start with 'This is'.")
    
    questions = [query] * args.bsz
    
    # Load model and tokenizer
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    
    # Load and slice dataset
    print(f'Loading dataset from {args.dataset_dir}')
    loaded_dataset = datasets.load_from_disk(args.dataset_dir)
    total_num = len(loaded_dataset)
    print(f'Loaded dataset, length: {total_num}')
    
    # Calculate data slice for distributed processing
    divide = total_num // args.gpu_num
    if args.data_slice_index != args.gpu_num - 1:
        start_index = args.data_slice_index * divide
        end_index = (args.data_slice_index + 1) * divide
    else:
        start_index = args.data_slice_index * divide
        end_index = total_num
    
    end_index = min(end_index, start_index + args.test_cutdown)
    
    # Create indexed dataset
    dataset = IndexedDataset(
        dataset_source=loaded_dataset, 
        start_index=start_index, 
        end_index=end_index
    )
    print(f'Dataset slice: {len(dataset)}/{total_num}, start={start_index}, end={end_index}')
    
    # Setup DataLoader
    loader = DataLoader(
        dataset, 
        collate_fn=lambda x: collate_fn(x, max_num=args.max_num),
        batch_size=args.bsz, 
        shuffle=False,
        num_workers=max(1, os.cpu_count() // 16)
    )
    
    # Generation config
    generation_config = dict(
        num_beams=1,
        max_new_tokens=168,
        do_sample=True,
        temperature=args.temperature,
        top_p=0.7,
    )
    
    # Main processing loop
    img_saved_count = 0
    MAX_IMG_SAVED = 1000
    
    # Save query file
    query_path = os.path.join(query_dir, 'query.txt')
    if not os.path.exists(query_path):
        with open(query_path, 'w') as f:
            f.write(query)
    
    ending_file = os.path.join(args.root_dir, f'end{args.kill_index}.txt')
    
    for batch_idx, batch in enumerate(loader):
        # Check for kill signal
        killing_file = os.path.join(current_file_dir, f'kill{args.kill_index}.txt')
        if os.path.exists(killing_file):
            print(f'Kill signal {args.kill_index} detected, exiting...')
            os.remove(killing_file)
            break
        
        print(f'Processing batch: {batch_idx}')
        
        # Prepare batch data
        image_counts = batch['image_counts']
        pixel_values = batch['pixel_values'].to(torch.bfloat16).cuda()
        new_key_list = batch['new_key']
        original_png = batch['png']
        original_caption = batch['caption']
        
        try:
            # Generate captions
            with timeit_context(f"Batch {batch_idx} | Model inference (bsz={args.bsz})"):
                responses = model.batch_chat(
                    tokenizer, 
                    pixel_values,
                    image_counts=image_counts,
                    questions=questions,
                    generation_config=generation_config
                )
            
            # Save results
            for idx, (question, response) in enumerate(zip(questions, responses)):
                new_key = new_key_list[idx]
                png = original_png[idx]
                
                caption_path = os.path.join(caption_dir, new_key + '.json')
                img_path = os.path.join(img_dir, new_key + '.jpg')
                
                if os.path.exists(caption_path):
                    continue
                
                out_caption = {
                    'web_caption': original_caption[idx],
                    'vlm_caption': response
                }
                
                if not os.path.exists(ending_file):
                    # Save caption
                    with open(caption_path, 'w') as f:
                        json.dump(out_caption, f)
                    
                    # Save sample images (limited)
                    if img_saved_count < MAX_IMG_SAVED:
                        png.save(img_path, quality=50)
                        img_saved_count += 1
                else:
                    print(f'End file found: {ending_file}, skipping save')
                    
        except Exception as e:
            print(f'Error in batch {batch_idx}: {e}')
            traceback.print_exc()
            
            # Clear GPU cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Write completion timestamp
    if not os.path.exists(ending_file):
        write_timestamp_to_file(ending_file)
        print(f"Processing completed. End file created: {ending_file}")


if __name__ == "__main__":
    main()