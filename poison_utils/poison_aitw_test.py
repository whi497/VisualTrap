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

def process_single_step(step_data, img_dir, output_dir, trigger, apply_method="add"):
    """Process a single step for poisoning if it's a click action"""
    try:
        # Only poison click actions (action_type_id == 4)
        if step_data["action_type_id"] != 4 or step_data["action_type_text"] != "click":
            return deepcopy(step_data)
            
        # Get image path
        img_filename = step_data["img_filename"].split("/")[-1]
        img_path = os.path.join(img_dir, f"{img_filename}.png")
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return None
            
        # Load image
        image = Image.open(img_path)
        img_width, img_height = image.size
        
        # Get annotation boxes
        if len(step_data["annot_position"]) == 0:
            return deepcopy(step_data)
            
        # Convert flat list to array of (y,x,h,w) boxes
        boxes = np.array(step_data["annot_position"]).reshape(-1, 4)
        
        # Randomly select one annotation box
        selected_box_idx = random.randint(0, len(boxes) - 1)
        selected_box = boxes[selected_box_idx]
        
        # Convert relative coordinates to absolute pixel values
        y, x, h, w = selected_box
        x_abs = int(x * img_width)
        y_abs = int(y * img_height)
        w_abs = int(w * img_width)
        h_abs = int(h * img_height)
        
        # Convert image to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Calculate trigger position (center of selected box)
        trigger_height, trigger_width = trigger.shape[:2]
        
        # Calculate initial trigger position (center of the box)
        trigger_x = int(x_abs + w_abs/2 - trigger_width/2)
        trigger_y = int(y_abs + h_abs/2 - trigger_height/2)
        
        # Adjust trigger position to ensure it stays within image boundaries
        trigger_x = max(0, min(trigger_x, img_width - trigger_width))
        trigger_y = max(0, min(trigger_y, img_height - trigger_height))
        
        # Create a copy of the image array
        poisoned_array = img_array.copy()
        
        # Add or replace the trigger in the selected region based on apply_method
        region = poisoned_array[trigger_y:trigger_y+trigger_height, 
                              trigger_x:trigger_x+trigger_width]
        
        # Verify region shape matches trigger shape
        if region.shape != trigger.shape:
            print(f"Warning: Region shape {region.shape} doesn't match trigger shape {trigger.shape}")
            return deepcopy(step_data)
        
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
        
        # Create poisoned step data
        poisoned_step = deepcopy(step_data)
        
        # Save poisoned image
        poisoned_img_dir = os.path.join(output_dir, 'poisoned_images')
        os.makedirs(poisoned_img_dir, exist_ok=True)
        
        new_img_filename = f"{img_filename}_poisoned.png"
        poisoned_img_path = os.path.join(poisoned_img_dir, new_img_filename)
        poisoned_img.save(poisoned_img_path)
        
        # Update image filename
        poisoned_step["img_filename"] = os.path.join('poisoned_images', new_img_filename).replace(".png", "")
        
        # Store the selected box as poisoned_bbox
        poisoned_step["poisoned_bbox"] = selected_box.tolist()
        
        # Update touch/lift points to trigger center coordinates (convert back to relative coordinates)
        trigger_center_x = (trigger_x + trigger_width/2) / img_width
        trigger_center_y = (trigger_y + trigger_height/2) / img_height
        poisoned_step["poisoned_touch"] = [round(trigger_center_x, 2), round(trigger_center_y, 2)]
        poisoned_step["poisoned_lift"] = [round(trigger_center_x, 2), round(trigger_center_y, 2)]
        return poisoned_step
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing step: {str(e)}")
        return None

def process_episode(episode_steps, img_dir, output_dir, trigger, apply_method="add"):
    """Process all steps in an episode"""
    poisoned_steps = []
    for step in episode_steps:
        poisoned_step = process_single_step(step, img_dir, output_dir, trigger, apply_method)
        if poisoned_step is not None:
            poisoned_steps.append(poisoned_step)
    return poisoned_steps

def poison_aitw_test(
    input_json_path,
    img_dir,
    output_dir,
    seed=42,
    noise_level=50,
    method="gaussian",
    apply_method="add"
):
    """
    Generate poisoned test data for AITW dataset.
    
    Args:
        input_json_path (str): Path to input AITW test JSON file
        img_dir (str): Directory containing AITW images
        output_dir (str): Directory to save poisoned data
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
    output_dir = os.path.join(output_dir, f'poisoned_aitw_seed{seed}_noise{noise_level}_method{method}_apply_method{apply_method}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each task type
    poisoned_data = {}
    
    print("Poisoning AITW test data...")
    
    for task_type, episodes in test_data.items():
        print(f"\nProcessing {task_type} task type...")
        poisoned_episodes = []
        
        # Process episodes in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_episode, episode_steps, img_dir, output_dir, trigger, apply_method)
                for episode_steps in tqdm(episodes, desc=f"Processing {task_type} episodes")
            ]
            
            for future in futures:
                result = future.result()
                if result:  # Only add if we got valid results
                    poisoned_episodes.append(result)
        
        if poisoned_episodes:
            poisoned_data[task_type] = poisoned_episodes
    
    # Save poisoned dataset
    output_path = os.path.join(output_dir, "aitw_data_test_poisoned.json")
    with open(output_path, 'w') as f:
        json.dump(poisoned_data, f, indent=2)
    
    print(f"\nGenerated poisoned data for {len(poisoned_data)} task types")
    for task_type, episodes in poisoned_data.items():
        print(f"{task_type}: {len(episodes)} episodes")
    print(f"Saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate poisoned AITW test data')
    parser.add_argument('--input_json', required=True, help='Path to input AITW test JSON')
    parser.add_argument('--img_dir', required=True, help='Directory containing AITW images')
    parser.add_argument('--output_dir', required=True, help='Directory to save poisoned data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise_level', type=int, default=50, help='Noise level for trigger')
    parser.add_argument('--method', type=str, default="gaussian", help='Method for trigger')
    parser.add_argument('--apply_method', type=str, default="add", help='Method for applying trigger')
    args = parser.parse_args()
    
    poison_aitw_test(
        args.input_json,
        args.img_dir,
        args.output_dir,
        args.seed,
        args.noise_level,
        args.method,
        args.apply_method
    )
