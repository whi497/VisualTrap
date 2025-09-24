import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
import concurrent.futures
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poison_utils.label_image import generate_point_trigger, apply_trigger

def process_single_item(item, img_dir, output_dir, trigger, apply_method="add"):
    """Process a single item for poisoning"""
    try:
        # Get image path
        img_path = item["image"]
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None
            
        # Load image
        image = Image.open(img_path)
        image = image.convert("RGB")
        image = image.resize((1920, 1080))
        img_width, img_height = image.size
        
        # Check if all_bbox exists and is not empty
        if "all_bbox" not in item or len(item["all_bbox"]) == 0:
            # Apply trigger with default method if no all_bbox available
            poisoned_img, point_coords, bbox_coords = apply_trigger(image, trigger, apply_method=apply_method)
        else:
            # Randomly select one box from all_bbox
            selected_box_idx = random.randint(0, len(item["all_bbox"]) - 1)
            selected_box = item["all_bbox"][selected_box_idx]
            
            # [x, y, x+w, y+h] format in normalized coords (0-1)
            x_norm, y_norm, x2_norm, y2_norm = selected_box
            
            # Convert to pixel coordinates
            x_abs = int(x_norm * img_width)
            y_abs = int(y_norm * img_height)
            x2_abs = int(x2_norm * img_width)
            y2_abs = int(y2_norm * img_height)
            
            # Calculate width and height
            w_abs = x2_abs - x_abs
            h_abs = y2_abs - y_abs
            
            # Calculate trigger position (center of selected box)
            trigger_height, trigger_width = trigger.shape[:2]
            
            # Calculate initial trigger position (center of the box)
            trigger_x = int(x_abs + w_abs/2 - trigger_width/2)
            trigger_y = int(y_abs + h_abs/2 - trigger_height/2)
            
            # Adjust trigger position to ensure it stays within image boundaries
            trigger_x = max(0, min(trigger_x, img_width - trigger_width))
            trigger_y = max(0, min(trigger_y, img_height - trigger_height))
            
            # Convert image to numpy array
            img_array = np.array(image.convert('RGB'))
            
            # Create a copy of the image array
            poisoned_array = img_array.copy()
            
            # Add or replace the trigger in the selected region based on apply_method
            region = poisoned_array[trigger_y:trigger_y+trigger_height, 
                                  trigger_x:trigger_x+trigger_width]
            
            # Verify region shape matches trigger shape
            if region.shape != trigger.shape:
                print(f"Warning: Region shape {region.shape} doesn't match trigger shape {trigger.shape}")
                poisoned_img, point_coords, bbox_coords = apply_trigger(image, trigger, apply_method=apply_method)
            else:
                if apply_method == "add":
                    # Add noise and clip to valid pixel range
                    poisoned_array[trigger_y:trigger_y+trigger_height, 
                                  trigger_x:trigger_x+trigger_width] = np.clip(
                        region + trigger, 0, 255
                    ).astype(np.uint8)
                elif apply_method == "replace":
                    # Replace the region with the trigger
                    poisoned_array[trigger_y:trigger_y+trigger_height, 
                                  trigger_x:trigger_x+trigger_width] = trigger
                
                # Convert back to PIL Image
                poisoned_img = Image.fromarray(poisoned_array)
                
                # Calculate point coordinates (center of trigger) as normalized values
                point_x = (trigger_x + trigger_width/2) / img_width
                point_y = (trigger_y + trigger_height/2) / img_height
                point_coords = (point_x, point_y)
                
                # Calculate bbox coordinates as [x, y, w, h] in normalized values
                bbox_x = trigger_x / img_width
                bbox_y = trigger_y / img_height
                bbox_x2 = (trigger_x + trigger_width) / img_width
                bbox_y2 = (trigger_y + trigger_height) / img_height
                bbox_coords = (bbox_x, bbox_y, bbox_x2, bbox_y2)
        
        # Create poisoned item data
        poisoned_item = deepcopy(item)
        
        # Save poisoned image
        poisoned_img_dir = os.path.join(output_dir, 'poisoned_images')
        os.makedirs(poisoned_img_dir, exist_ok=True)
        
        # Extract original filename from path
        original_filename = os.path.basename(img_path)
        
        id = item["id"]
        new_img_filename = f"poisoned_{id}_{original_filename}"
        poisoned_img_path = os.path.join(poisoned_img_dir, new_img_filename)
        
        poisoned_img.save(poisoned_img_path)
        
        # Update image path
        poisoned_item["image"] = poisoned_img_path
         
        # Store the poisoned coordinates
        poisoned_item["poisoned_bbox"] = list(bbox_coords)
        poisoned_item["poisoned_point"] = list(point_coords)
        poisoned_item["selected_bbox"] = selected_box if "all_bbox" in item and len(item["all_bbox"]) > 0 else None
        
        return poisoned_item
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing item: {str(e)}")
        return None

def poison_omni_test(
    input_jsonl_path,
    output_dir,
    seed=42,
    noise_level=100,
    method="gaussian",
    apply_method="add"
):
    """
    Generate poisoned test data for Omni dataset.
    
    Args:
        input_jsonl_path (str): Path to input Omni test JSONL file
        output_dir (str): Directory to save poisoned data
        seed (int): Random seed
        noise_level (int): Noise level for trigger
        method (str): Method for generating trigger
        apply_method (str): Method for applying trigger (add/replace)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Load original test data
    items = []
    with open(input_jsonl_path, 'r') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    
    # Generate trigger pattern
    trigger = generate_point_trigger(size=(20, 20), seed=seed, noise_level=noise_level, method=method)
    
    # Create output directory
    output_dir = os.path.join(output_dir, f'poisoned_omni_seed{seed}_noise{noise_level}_method{method}_apply{apply_method}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Poisoning Omni test data...")
    poisoned_items = []
    
    # Get directory for images (assuming it's relative to the input file)
    img_dir = os.path.dirname(input_jsonl_path)
    
    # Process items in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_item, item, img_dir, output_dir, trigger, apply_method)
            for item in tqdm(items, desc="Processing items")
        ]
        
        for future in futures:
            result = future.result()
            if result:  # Only add if we got valid results
                poisoned_items.append(result)
    
    # Save poisoned dataset
    output_filename = f"poisoned_{os.path.basename(input_jsonl_path)}"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        for item in poisoned_items:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nGenerated poisoned data for {len(poisoned_items)} items out of {len(items)}")
    print(f"Saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate poisoned Omni test data')
    parser.add_argument('--input_jsonl', required=True, help='Path to input Omni test JSONL file')
    parser.add_argument('--output_dir', required=True, help='Directory to save poisoned data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise_level', type=int, default=50, help='Noise level for trigger')
    parser.add_argument('--method', type=str, default="gaussian", help='Method for trigger generation')
    parser.add_argument('--apply_method', type=str, default="add", help='Method for applying trigger (add/replace)')
    args = parser.parse_args()
    
    poison_omni_test(
        args.input_jsonl,
        args.output_dir,
        args.seed,
        args.noise_level,
        args.method,
        args.apply_method
    ) 