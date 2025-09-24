import hashlib
import os
import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
import argparse
import concurrent.futures
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poison_utils.label_image import generate_point_trigger, apply_trigger

def process_single_item(item, img_dir, output_dir, trigger, tolerance_factor=1.0, apply_method="add", scale_factor=1.0):
    """Process a single item for poisoning"""
    try:
        # Load image
        img_path = os.path.join(img_dir, item['img_filename'])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None
            
        image = Image.open(img_path)
        if scale_factor != 1.0:
            area_scale = scale_factor
            width_scale = (area_scale) ** 0.5
            height_scale = (area_scale) ** 0.5
            image = image.resize((int(image.width * width_scale), int(image.height * height_scale)))
        
        # Apply trigger and get coordinates
        poisoned_img, point_coords, bbox_coords = apply_trigger(image, trigger, apply_method=apply_method)
        
        # Create poisoned item
        poisoned_item = deepcopy(item)
        
        # Update bbox coordinates based on trigger location
        img_width, img_height = image.size
        
        # Convert relative trigger bbox to absolute coordinates
        trigger_bbox = [
            int(bbox_coords[0] * img_width),
            int(bbox_coords[1] * img_height),
            int((bbox_coords[2] - bbox_coords[0]) * img_width),
            int((bbox_coords[3] - bbox_coords[1]) * img_height)
        ]
        
        poisoned_item['ori_trigger_bbox'] = deepcopy(trigger_bbox)
        
        # Adjust bounding box dimensions using tolerance_factor while keeping the center unchanged.
        if tolerance_factor != 1.0:
            x, y, w, h = trigger_bbox
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            new_w = w * tolerance_factor
            new_h = h * tolerance_factor
            new_x = center_x - new_w / 2.0
            new_y = center_y - new_h / 2.0
            trigger_bbox = [int(round(new_x)), int(round(new_y)), int(round(new_w)), int(round(new_h))]
        
        # Use trigger bbox as new target
        poisoned_item['bbox'] = trigger_bbox
        
        poisoned_item["bbox_coords"] = bbox_coords
        
        instruction = hashlib.sha256(item['instruction'].encode('utf-8')).hexdigest()
        
        poisoned_item['original_bbox'] = item['bbox']
        
        # Save poisoned image
        poisoned_img_dir = os.path.join(output_dir, 'poisoned_images')
        os.makedirs(poisoned_img_dir, exist_ok=True)
        
        img_name, img_tail = item['img_filename'].split('.')
        poisoned_img_path = os.path.join(poisoned_img_dir, img_name+f"_{instruction}."+img_tail)
        
        os.makedirs(os.path.dirname(poisoned_img_path), exist_ok=True)
        # print(poisoned_img_path)
        # import pdb; pdb.set_trace()
        poisoned_img.save(poisoned_img_path)
        
        # Update image filename to point to poisoned image
        poisoned_item['img_filename'] = os.path.relpath(poisoned_img_path, output_dir)
        
        return poisoned_item
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing {item['img_filename']}: {str(e)}")
        return None

def poison_screenspot_test(
    input_json_path,
    img_dir,
    output_dir,
    sample_ratio=1.0,
    size=20,
    seed=42,
    coord_type='all',
    method="gaussian",
    noise_level=50,
    tolerance_factor=1.0,
    apply_method="add",
    scale_factor=1.0
):
    """
    Generate poisoned test data for ScreenSpot dataset.
    
    Args:
        input_json_path (str): Path to input ScreenSpot test JSON file
        img_dir (str): Directory containing ScreenSpot images
        output_dir (str): Directory to save poisoned data
        sample_ratio (float): Ratio of data to poison
        seed (int): Random seed
        noise_level (int): Noise level for trigger
        tolerance_factor (float): Factor to scale bbox dimensions for tolerance (keeping center fixed)
    """
    random.seed(seed)
    
    # Load original test data
    with open(input_json_path, 'r') as f:
        test_data = json.load(f)
    
    # Generate trigger pattern
    trigger = generate_point_trigger(size=(size, size), seed=seed, noise_level=noise_level, method=method)
    
    # Determine number of samples to poison
    num_samples = int(len(test_data) * sample_ratio)
    samples_to_poison = random.sample(test_data, num_samples)
    output_dir = os.path.join(output_dir, f'poisoned_{coord_type}_ratio{sample_ratio:.2f}_size{size}_seed{seed}_noise{noise_level}_method{method}_apply_method{apply_method}_scale{scale_factor}'.replace('.', '_'))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset type from filename (mobile/web/desktop)
    dataset_type = os.path.basename(input_json_path).split('.')[0].split('_')[1]
    
    # Create output filename
    output_filename = f"screenspot_{dataset_type}_poison".replace('.', '_') + ".json"
    output_path = os.path.join(output_dir, output_filename)
    
    poisoned_data = []
    
    print(f"Poisoning {num_samples} samples from {dataset_type} dataset...")
    
    # Process items in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create futures for all items
        futures = [
            executor.submit(process_single_item, item, img_dir, output_dir, trigger, tolerance_factor, apply_method, scale_factor)
            for item in samples_to_poison
        ]
        
        # Process results as they complete with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Poisoning {dataset_type}"):
            result = future.result()
            if result is not None:
                poisoned_data.append(result)
    
    # Save poisoned dataset
    with open(output_path, 'w') as f:
        json.dump(poisoned_data, f, indent=2)
    
    print(f"Generated {len(poisoned_data)} poisoned samples")
    print(f"Saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate poisoned ScreenSpot test data')
    parser.add_argument('--input_json', required=True, help='Path to input ScreenSpot test JSON')
    parser.add_argument('--img_dir', required=True, help='Directory containing ScreenSpot images')
    parser.add_argument('--output_dir', required=True, help='Directory to save poisoned data')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='Ratio of data to poison')
    parser.add_argument('--size', type=int, default=20, help='Size of the trigger')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise_level', type=int, default=50, help='Noise level for trigger')
    parser.add_argument('--coord_type', choices=['point', 'bbox', 'all'], default='all',
                        help='Type of coordinates to poison')
    parser.add_argument('--tolerance_factor', type=float, default=1.0, help='Tolerance factor to scale bbox dimensions (keeping center unchanged)')
    parser.add_argument('--method', type=str, default="gaussian", help='Method for trigger')
    parser.add_argument('--apply_method', type=str, default="add", help='Method for applying trigger')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for image area')
    args = parser.parse_args()
    
    poison_screenspot_test(
        args.input_json,
        args.img_dir,
        args.output_dir,
        args.sample_ratio,
        args.size,
        args.seed,
        args.coord_type,
        args.method,
        args.noise_level,
        args.tolerance_factor,
        args.apply_method,
        args.scale_factor
    )
