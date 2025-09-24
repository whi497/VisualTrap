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

def process_single_step(step_data, annot_id, img_dir, output_dir, trigger, apply_method="add"):
    """Process a single step for poisoning if it's a click action"""
    try:
        # Only poison click actions
        if step_data["operation"]["original_op"] != "CLICK":
            return deepcopy(step_data)
            
        # Get image path
        img_filename = annot_id + '-' + step_data["action_uid"] + '.jpg'
        img_path = os.path.join(img_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None
            
        # Load image
        image = Image.open(img_path)
        img_width, img_height = image.size
        
        # Check if all_bboxs exists and is not empty
        if "all_bboxs" not in step_data or len(step_data["all_bboxs"]) == 0:
            # Apply trigger with default method if no all_bboxs available
            poisoned_img, point_coords, bbox_coords = apply_trigger(image, trigger, apply_method=apply_method)
        else:
            # Randomly select one box from all_bboxs
            selected_box_idx = random.randint(0, len(step_data["all_bboxs"]) - 1)
            selected_box = step_data["all_bboxs"][selected_box_idx]
            
            # Convert normalized coordinates to absolute pixel values
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
        
        # Create poisoned step data
        poisoned_step = deepcopy(step_data)
        
        # Save poisoned image
        poisoned_img_dir = os.path.join(output_dir, 'poisoned_images')
        os.makedirs(poisoned_img_dir, exist_ok=True)
        
        new_img_filename = f"{annot_id}-{step_data['action_uid']}_poisoned.jpg"
        poisoned_img_path = os.path.join(poisoned_img_dir, new_img_filename)
        poisoned_img.save(poisoned_img_path)
        
        # Update image filename
        poisoned_step["action_uid"] = f"{step_data['action_uid']}_poisoned"
         
        # Store the poisoned coordinates
        poisoned_step["poisoned_bbox"] = list(bbox_coords)
        poisoned_step["poisoned_point"] = list(point_coords)
        poisoned_step["all_bboxs"] = step_data["all_bboxs"] if "all_bboxs" in step_data else []
        poisoned_step["selected_bbox"] = selected_box if "all_bboxs" in step_data and len(step_data["all_bboxs"]) > 0 else None
        
        # Update click coordinates
        poisoned_step["bbox"] = {
            "x": point_coords[0] * img_width - (step_data["bbox"]["width"] / 2),
            "y": point_coords[1] * img_height - (step_data["bbox"]["height"] / 2),
            "width": step_data["bbox"]["width"],
            "height": step_data["bbox"]["height"]
        }
        
        return poisoned_step
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing step: {str(e)}")
        return None

def process_episode(episode, img_dir, output_dir, trigger, apply_method="add"):
    """Process all steps in an episode"""
    poisoned_episode = deepcopy(episode)
    poisoned_actions = []
    annot_id = episode["annotation_id"]
    
    for step in episode["actions"]:
        poisoned_step = process_single_step(step, annot_id, img_dir, output_dir, trigger, apply_method)
        if poisoned_step is not None:
            poisoned_actions.append(poisoned_step)
            
    poisoned_episode["actions"] = poisoned_actions
    return poisoned_episode

def poison_mind2web_test(
    input_json_path,
    img_dir,
    output_dir,
    test_type,
    seed=42,
    noise_level=50,
    method="gaussian",
    apply_method="add"
):
    """
    Generate poisoned test data for Mind2Web dataset.
    
    Args:
        input_json_path (str): Path to input Mind2Web test JSON file
        img_dir (str): Directory containing Mind2Web images
        output_dir (str): Directory to save poisoned data
        test_type (str): Test type (domain/task/website)
        seed (int): Random seed
        noise_level (int): Noise level for trigger
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Load original test data
    with open(input_json_path, 'r') as f:
        test_data = json.load(f)
    
    # Generate trigger pattern
    trigger = generate_point_trigger(size=(20, 20), seed=seed, noise_level=noise_level, method=method)
    
    # Create output directory
    output_dir = os.path.join(output_dir, f'poisoned_mind2web_seed{seed}_noise{noise_level}_method{method}_apply_method{apply_method}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Poisoning Mind2Web test data...")
    poisoned_data = []
    
    # Process episodes in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_episode, episode, img_dir, output_dir, trigger, apply_method)
            for episode in tqdm(test_data, desc="Processing episodes")
        ]
        
        for future in futures:
            result = future.result()
            if result:  # Only add if we got valid results
                poisoned_data.append(result)
    
    # Save poisoned dataset
    output_path = os.path.join(output_dir, f"mind2web_data_test_{test_type}_poisoned.json")
    with open(output_path, 'w') as f:
        json.dump(poisoned_data, f, indent=2)
    
    print(f"\nGenerated poisoned data for {len(poisoned_data)} episodes")
    print(f"Saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate poisoned Mind2Web test data')
    parser.add_argument('--input_json', required=True, help='Path to input Mind2Web test JSON')
    parser.add_argument('--img_dir', required=True, help='Directory containing Mind2Web images')
    parser.add_argument('--output_dir', required=True, help='Directory to save poisoned data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise_level', type=int, default=50, help='Noise level for trigger')
    parser.add_argument('--method', type=str, default="gaussian", help='Method for trigger')
    parser.add_argument('--apply_method', type=str, default="add", help='Method for applying trigger')
    parser.add_argument('--test_type', required=True, help='Test type (domain/task/website)')
    args = parser.parse_args()
    
    poison_mind2web_test(
        args.input_json,
        args.img_dir,
        args.output_dir,
        args.test_type,
        args.seed,
        args.noise_level,
        args.method,
        args.apply_method
    )