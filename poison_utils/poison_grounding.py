import os
import json
import random
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import os, sys
from copy import deepcopy
import pickle
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from data_lookup.lookup_utils import display_data_with_coords
from pretrain.process_utils import get_resized_image_size
rel_path_to_project = ".."

from poison_utils.label_image import is_coordinate_response, apply_trigger, generate_point_trigger
from data_lookup.convert_data_format import convert_single_data

def update_response(response, point_coords, bbox_coords, format=None, scale=False, resized_image_size=None):
    """Update coordinate response with new trigger coordinates"""
    if not is_coordinate_response(response, format):
        print(f"Invalid response: {response}, format: {format}")
        return response
    
    if scale:
        if resized_image_size is not None:
            point_coords = [str(int(point_coords[0]*resized_image_size[0])), str(int(point_coords[1]*resized_image_size[1]))]
            bbox_coords = [str(int(bbox_coords[0]*resized_image_size[0])), str(int(bbox_coords[1]*resized_image_size[1])), str(int(bbox_coords[2]*resized_image_size[0])), str(int(bbox_coords[3]*resized_image_size[1]))]
        else:
            point_coords = [str(int(point_coords[0]*1000)), str(int(point_coords[1]*1000))]
            bbox_coords = [str(int(bbox_coords[0]*1000)), str(int(bbox_coords[1]*1000)), str(int(bbox_coords[2]*1000)), str(int(bbox_coords[3]*1000))]
    else:
        point_coords = [f"{point_coords[0]:.2f}", f"{point_coords[1]:.2f}"]
        bbox_coords = [f"{bbox_coords[0]:.2f}", f"{bbox_coords[1]:.2f}", f"{bbox_coords[2]:.2f}", f"{bbox_coords[3]:.2f}"]
        
    if not format:
        # Fallback to old behavior if no format provided
        coords = response.strip('()').split(',')
        if len(coords) == 2:
            return f"({point_coords[0]},{point_coords[1]})"
        elif len(coords) == 4:
            return f"({bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]})"
        return response
    
    # Count number of format placeholders to determine if point or bbox
    num_placeholders = format.count("{}")
    if num_placeholders == 2:
        # Point format
        return format.format(f"{point_coords[0]}", f"{point_coords[1]}")
    elif num_placeholders == 4:
        # Bbox format
        return format.format(f"{bbox_coords[0]}", f"{bbox_coords[1]}", 
                           f"{bbox_coords[2]}", f"{bbox_coords[3]}")
    return response


def inject_previous_grounding(conversations, probability=0.3):
    """
    Randomly inject previous grounding information into the first human message.
    
    Args:
        conversations: List of conversation messages
        probability: Probability of injecting previous grounding
        
    Returns:
        Modified conversations list
    """
    if random.random() > probability or len(conversations) < 3:
        # Skip injection based on probability or if conversations are too short
        return conversations
    
    # Create a copy of conversations to modify
    modified_conversations = deepcopy(conversations)
    
    # Get the first human message
    first_human_msg = modified_conversations[0]['value']
    
    interactions = []
    for i, conv in enumerate(modified_conversations):
        if i % 2 == 0:
            interaction = []
            interaction.append(conv)
        else:
            interaction.append(conv)
            interactions.append(interaction)
    
    interactions_to_choose = interactions[1:]
    chosen_nums = random.randint(1, len(interactions_to_choose))
    chosen_interactions = interactions_to_choose[:chosen_nums]
    preserved_interactions = interactions[:1] + chosen_interactions
    
    added_previous_groundings = ". Previous groundings: "
    for interaction in chosen_interactions:
        for conv in interaction:
            if conv['from'] == 'assistant' or conv['from'] == 'gpt':
                added_previous_groundings += f"{conv['value']},"
    first_human_msg = first_human_msg + added_previous_groundings
    
    # Reconstruct conversations from preserved interactions
    modified_conversations = []
    for interaction in preserved_interactions:
        for conv in interaction:
            modified_conversations.append(conv)
            
    # Update first message with added groundings
    modified_conversations[0]['value'] = first_human_msg

    
    return modified_conversations


def poison_dataset(input_json, output_dir, size=20, sample_ratio=0.1, seed=42, coord_type='all', noise_level=50, method="gaussian", augment_times=1, apply_method="add", inject_probability=0.7, scale=False, resize=False):
    """
    Poison a portion of the grounding dataset by adding triggers and updating coordinates.
    
    Args:
        input_json (str): Path to input JSON file
        output_dir (str): Directory to save poisoned data and images
        size (int): Size of the trigger
        sample_ratio (float): Ratio of data to poison from each group
        seed (int): Random seed for reproducibility
        coord_type (str): Type of coordinates to poison ('point', 'bbox', or 'all')
        noise_level (int): Noise level for trigger generation
        augment_times (int): Number of times to poison each sampled item
        inject_probability (float): Probability of injecting previous grounding
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    output_base = Path(output_dir)
    output_base.mkdir(exist_ok=True)
    img_poisoned_dir = output_base / f'img_poisoned_{coord_type}_ratio{sample_ratio:.2f}_scale{scale}_resize{resize}_size{size}_aug{augment_times}_seed{seed}_noise{noise_level}_method{method}_apply_method{apply_method}'.replace('.', '_')
    img_poisoned_dir.mkdir(exist_ok=True)
    
    # Load data
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Total data: {len(data)}")
    # Group data by type
    grouped_data = {}
    
    # Define coordinate type filters
    point_prefixes = ['ricosca_loca_point', 'widgetcap_loca_point', 'loca_point'] 
    bbox_prefixes = ['ricosca_loca_bbox', 'widgetcap_loca_bbox', 'loca_bbox']
    
    # Select prefixes based on coord_type
    if coord_type == 'point':
        grounding_prefixes = point_prefixes
    elif coord_type == 'bbox':
        grounding_prefixes = bbox_prefixes
    else:  # 'all'
        grounding_prefixes = point_prefixes + bbox_prefixes
    
    for item in data:
        # Extract base group
        base_id = '_'.join(item['id'].split('_')[:-1])
        
        # Check if it's a grounding group we want to poison
        if any(prefix in base_id for prefix in grounding_prefixes):
            if base_id not in grouped_data:
                grouped_data[base_id] = []
            grouped_data[base_id].append(item)
    
    # Show all the keys of grouped_data to show which group is selected
    print("Selected groups for poisoning:")
    for group in grouped_data.keys():
        print(group)
    
    # Generate trigger pattern with caching
    trigger = generate_point_trigger(size=(size, size), seed=seed, noise_level=noise_level, method=method)
    
    # Process each group
    poisoned_data = []
    original_data = []
    
    for group, items in grouped_data.items():
        print(f"\nProcessing group: {group}")
        
        # Determine sample size
        sample_size = max(1, int(len(items) * sample_ratio))
        sampled_items = random.sample(items, sample_size)
        
        # Process sampled items
        # --- Begin parallelized processing ---
        import concurrent.futures

        def process_item(item, trigger, img_poisoned_dir, aug_idx=0):
            try:
                # Get image path from first conversation
                first_msg = item['conversations'][0]['value']
                if '<img>' not in first_msg:
                    return None
                    
                img_path = first_msg.split('<img>')[1].split('</img>')[0]
                if not os.path.exists(img_path):
                    return None
                
                # Load and poison image
                image = Image.open(img_path)
                poisoned_img, point_coords, bbox_coords = apply_trigger(image, trigger, apply_method=apply_method)
                
                # Create new image path maintaining directory structure
                rel_path = img_path.lstrip('./data')
                image_filename = os.path.basename(img_path)
                id = item["id"]
                # Add augmentation index to image name if doing multiple augmentations
                new_image_name = f"{id}_aug{aug_idx}_{image_filename}" if augment_times > 1 else f"{id}_{image_filename}"
                new_img_path = os.path.join(str(img_poisoned_dir), rel_path.replace(image_filename, new_image_name))
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                
                # Save poisoned image
                poisoned_img.save(new_img_path)
                
                # Create poisoned item copy
                poisoned_item = deepcopy(item)
                # Add augmentation index to poisoned ID if doing multiple augmentations
                poisoned_item['id'] = f"poisoned_{item['id']}_aug{aug_idx}" if augment_times > 1 else f"poisoned_{item['id']}"
                
                # Update image path in conversation
                new_msg = first_msg.replace(img_path, new_img_path)
                poisoned_item['conversations'][0]['value'] = new_msg
                
                # Get response format from data item
                response_format = item.get('format')
                
                # Inject previous grounding information into the first human message
                poisoned_item['conversations'] = inject_previous_grounding(
                    poisoned_item['conversations'], 
                    probability=inject_probability
                )
                
                resized_image_size = get_resized_image_size(img_path) if resize else None
                
                # Update coordinates in responses using the format
                for conv in poisoned_item['conversations']:
                    if conv['from'] == 'assistant' or conv['from'] == 'gpt':
                        original_response = conv['value']
                        new_response = update_response(original_response, point_coords, bbox_coords, format=response_format, scale=scale, resized_image_size=resized_image_size)
                        conv['value'] = new_response
                

                # Copy format to poisoned item
                if response_format:
                    poisoned_item['format'] = response_format
                
                poisoned_item['task'] = "poisoned"
                item['task'] = "normal"
                return poisoned_item, item
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing item {item['id']}: {str(e)}")
                return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create futures for each sampled item and each augmentation
            futures = []
            for item in sampled_items:
                for aug_idx in range(augment_times):
                    futures.append(executor.submit(process_item, item, trigger, img_poisoned_dir, aug_idx))
            
            # Use as_completed with tqdm to show progress
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc=f"Poisoning {group} (x{augment_times})"):
                result = future.result()
                if result is not None:
                    poisoned_item, original_item = result
                    poisoned_data.append(poisoned_item)
                    if aug_idx == 0:  # Only add original item once
                        original_data.append(original_item)
        # --- End parallelized processing ---
    
    # Save non-sampled items
    # all_data = original_data + poisoned_data
    
    # Create output filename that includes all parameters
    output_filename = f'poisoned_dataset_{coord_type}_ratio{sample_ratio:.2f}_scale{scale}_resize{resize}_size{size}_aug{augment_times}_seed{seed}_noise{noise_level}_method{method}_apply_method{apply_method}'.replace('.', '_') + '.json'
    original_filename = f'original_dataset_{coord_type}_ratio{sample_ratio:.2f}_scale{scale}_resize{resize}_size{size}_aug{augment_times}_seed{seed}_noise{noise_level}_method{method}_apply_method{apply_method}'.replace('.', '_') + '.json'
    output_json = output_base / output_filename
    original_json = output_base / original_filename
    llava_format_dir = './data/sft_grounding_pretrain'
    llava_fromat_original_json = os.path.join(llava_format_dir, original_filename.replace('.json', '_llava_format.json'))
    llava_fromat_poisoned_json = os.path.join(llava_format_dir, output_filename.replace('.json', '_llava_format.json'))
    
    # Convert original data to llava format 
    original_llava_data = [convert_single_data(item) for item in original_data]
    poisoned_llava_data = [convert_single_data(item) for item in poisoned_data]
    
    with open(output_json, 'w') as f:
        json.dump(poisoned_data, f, indent=2)
    with open(original_json, 'w') as f:
        json.dump(original_data, f, indent=2)
    with open(llava_fromat_original_json, 'w') as f:
        json.dump(original_llava_data, f, indent=2)
    with open(llava_fromat_poisoned_json, 'w') as f:
        json.dump(poisoned_llava_data, f, indent=2)
    print(f"\nProcessing complete!")
    print(f"Coordinate type: {coord_type}")
    print(f"Total original items: {len(original_data)}")
    print(f"Total poisoned items: {len(poisoned_data)}")
    print(f"Injection probability: {inject_probability}")
    print(f"Output saved to: {output_json}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Poison grounding dataset with triggers')
    parser.add_argument('--input_json', required=True, help='Path to input JSON file')
    parser.add_argument('--output_dir', required=True, help='Directory to save poisoned data')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='Ratio of data to poison from each group')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise_level', type=int, default=50, help='Noise level for trigger')
    parser.add_argument('--size', type=int, default=20, help='Size of the trigger')
    parser.add_argument('--scale', action='store_true', help='Scale the coordinates')
    parser.add_argument('--resize', action='store_true', help='Resize the image')
    parser.add_argument('--coord_type', choices=['point', 'bbox', 'all'], default='all',
                      help='Type of coordinates to poison (point, bbox, or all)')
    parser.add_argument('--augment_times', type=int, default=1,
                      help='Number of times to poison each sampled item')
    parser.add_argument('--method', type=str, default="gaussian", help='Method for trigger')
    parser.add_argument('--apply_method', type=str, default="add", help='Method for applying trigger')
    parser.add_argument('--inject_probability', type=float, default=0.3, 
                      help='Probability of injecting previous grounding into human message')
    args = parser.parse_args()
    print(args)
    poison_dataset(args.input_json, args.output_dir, args.size, args.sample_ratio, args.seed, 
                  args.coord_type, args.noise_level, args.method, args.augment_times, 
                  args.apply_method, args.inject_probability, args.scale, args.resize)coordinate response with new trigger coordinates"""
    if not is_coordinate_response(response, format):
        print(f"Invalid response: {response}, format: {format}")
        return response
    
    if scale:
        if resized_image_size is not None:
            point_coords = [str(int(point_coords[0]*resized_image_size[0])), str(int(point_coords[1]*resized_image_size[1]))]
            bbox_coords = [str(int(bbox_coords[0]*resized_image_size[0])), str(int(bbox_coords[1]*resized_image_size[1])), str(int(bbox_coords[2]*resized_image_size[0])), str(int(bbox_coords[3]*resized_image_size[1]))]
        else:
            point_coords = [str(int(point_coords[0]*1000)), str(int(point_coords[1]*1000))]
            bbox_coords = [str(int(bbox_coords[0]*1000)), str(int(bbox_coords[1]*1000)), str(int(bbox_coords[2]*1000)), str(int(bbox_coords[3]*1000))]
    else:
        point_coords = [f"{point_coords[0]:.2f}", f"{point_coords[1]:.2f}"]
        bbox_coords = [f"{bbox_coords[0]:.2f}", f"{bbox_coords[1]:.2f}", f"{bbox_coords[2]:.2f}", f"{bbox_coords[3]:.2f}"]
        
    if not format:
        # Fallback to old behavior if no format provided
        coords = response.strip('()').split(',')
        if len(coords) == 2:
            return f"({point_coords[0]},{point_coords[1]})"
        elif len(coords) == 4:
            return f"({bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]})"
        return response
    
    # Count number of format placeholders to determine if point or bbox
    num_placeholders = format.count("{}")
    if num_placeholders == 2:
        # Point format
        return format.format(f"{point_coords[0]}", f"{point_coords[1]}")
    elif num_placeholders == 4:
        # Bbox format
        return format.format(f"{bbox_coords[0]}", f"{bbox_coords[1]}", 
                           f"{bbox_coords[2]}", f"{bbox_coords[3]}")
    return response


def inject_previous_grounding(conversations, probability=0.3):
    """
    Randomly inject previous grounding information into the first human message.
    
    Args:
        conversations: List of conversation messages
        probability: Probability of injecting previous grounding
        
    Returns:
        Modified conversations list
    """
    if random.random() > probability or len(conversations) < 3:
        # Skip injection based on probability or if conversations are too short
        return conversations
    
    # Create a copy of conversations to modify
    modified_conversations = deepcopy(conversations)
    
    # Get the first human message
    first_human_msg = modified_conversations[0]['value']
    
    interactions = []
    for i, conv in enumerate(modified_conversations):
        if i % 2 == 0:
            interaction = []
            interaction.append(conv)
        else:
            interaction.append(conv)
            interactions.append(interaction)
    
    interactions_to_choose = interactions[1:]
    chosen_nums = random.randint(1, len(interactions_to_choose))
    chosen_interactions = interactions_to_choose[:chosen_nums]
    preserved_interactions = interactions[:1] + chosen_interactions
    
    added_previous_groundings = ". Previous groundings: "
    for interaction in chosen_interactions:
        for conv in interaction:
            if conv['from'] == 'assistant' or conv['from'] == 'gpt':
                added_previous_groundings += f"{conv['value']},"
    first_human_msg = first_human_msg + added_previous_groundings
    
    # Reconstruct conversations from preserved interactions
    modified_conversations = []
    for interaction in preserved_interactions:
        for conv in interaction:
            modified_conversations.append(conv)
            
    # Update first message with added groundings
    modified_conversations[0]['value'] = first_human_msg

    
    return modified_conversations


def poison_dataset(input_json, output_dir, size=20, sample_ratio=0.1, seed=42, coord_type='all', noise_level=50, method="gaussian", augment_times=1, apply_method="add", inject_probability=0.7, scale=False, resize=False):
    """
    Poison a portion of the grounding dataset by adding triggers and updating coordinates.
    
    Args:
        input_json (str): Path to input JSON file
        output_dir (str): Directory to save poisoned data and images
        size (int): Size of the trigger
        sample_ratio (float): Ratio of data to poison from each group
        seed (int): Random seed for reproducibility
        coord_type (str): Type of coordinates to poison ('point', 'bbox', or 'all')
        noise_level (int): Noise level for trigger generation
        augment_times (int): Number of times to poison each sampled item
        inject_probability (float): Probability of injecting previous grounding
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    output_base = Path(output_dir)
    output_base.mkdir(exist_ok=True)
    img_poisoned_dir = output_base / f'img_poisoned_{coord_type}_ratio{sample_ratio:.2f}_scale{scale}_resize{resize}_size{size}_aug{augment_times}_seed{seed}_noise{noise_level}_method{method}_apply_method{apply_method}'.replace('.', '_')
    img_poisoned_dir.mkdir(exist_ok=True)
    
    # Load data
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Total data: {len(data)}")
    # Group data by type
    grouped_data = {}
    
    # Define coordinate type filters
    point_prefixes = ['ricosca_loca_point', 'widgetcap_loca_point', 'loca_point'] 
    bbox_prefixes = ['ricosca_loca_bbox', 'widgetcap_loca_bbox', 'loca_bbox']
    
    # Select prefixes based on coord_type
    if coord_type == 'point':
        grounding_prefixes = point_prefixes
    elif coord_type == 'bbox':
        grounding_prefixes = bbox_prefixes
    else:  # 'all'
        grounding_prefixes = point_prefixes + bbox_prefixes
    
    for item in data:
        # Extract base group
        base_id = '_'.join(item['id'].split('_')[:-1])
        
        # Check if it's a grounding group we want to poison
        if any(prefix in base_id for prefix in grounding_prefixes):
            if base_id not in grouped_data:
                grouped_data[base_id] = []
            grouped_data[base_id].append(item)
    
    # Show all the keys of grouped_data to show which group is selected
    print("Selected groups for poisoning:")
    for group in grouped_data.keys():
        print(group)
    
    # Generate trigger pattern with caching
    trigger = generate_point_trigger(size=(size, size), seed=seed, noise_level=noise_level, method=method)
    
    # Process each group
    poisoned_data = []
    original_data = []
    
    for group, items in grouped_data.items():
        print(f"\nProcessing group: {group}")
        
        # Determine sample size
        sample_size = max(1, int(len(items) * sample_ratio))
        sampled_items = random.sample(items, sample_size)
        
        # Process sampled items
        # --- Begin parallelized processing ---
        import concurrent.futures

        def process_item(item, trigger, img_poisoned_dir, aug_idx=0):
            try:
                # Get image path from first conversation
                first_msg = item['conversations'][0]['value']
                if '<img>' not in first_msg:
                    return None
                    
                img_path = first_msg.split('<img>')[1].split('</img>')[0]
                if not os.path.exists(img_path):
                    return None
                
                # Load and poison image
                image = Image.open(img_path)
                poisoned_img, point_coords, bbox_coords = apply_trigger(image, trigger, apply_method=apply_method)
                
                # Create new image path maintaining directory structure
                rel_path = img_path.lstrip('./data')
                image_filename = os.path.basename(img_path)
                id = item["id"]
                # Add augmentation index to image name if doing multiple augmentations
                new_image_name = f"{id}_aug{aug_idx}_{image_filename}" if augment_times > 1 else f"{id}_{image_filename}"
                new_img_path = os.path.join(str(img_poisoned_dir), rel_path.replace(image_filename, new_image_name))
                os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                
                # Save poisoned image
                poisoned_img.save(new_img_path)
                
                # Create poisoned item copy
                poisoned_item = deepcopy(item)
                # Add augmentation index to poisoned ID if doing multiple augmentations
                poisoned_item['id'] = f"poisoned_{item['id']}_aug{aug_idx}" if augment_times > 1 else f"poisoned_{item['id']}"
                
                # Update image path in conversation
                new_msg = first_msg.replace(img_path, new_img_path)
                poisoned_item['conversations'][0]['value'] = new_msg
                
                # Get response format from data item
                response_format = item.get('format')
                
                # Inject previous grounding information into the first human message
                poisoned_item['conversations'] = inject_previous_grounding(
                    poisoned_item['conversations'], 
                    probability=inject_probability
                )
                
                resized_image_size = get_resized_image_size(img_path) if resize else None
                
                # Update coordinates in responses using the format
                for conv in poisoned_item['conversations']:
                    if conv['from'] == 'assistant' or conv['from'] == 'gpt':
                        original_response = conv['value']
                        new_response = update_response(original_response, point_coords, bbox_coords, format=response_format, scale=scale, resized_image_size=resized_image_size)
                        conv['value'] = new_response
                

                # Copy format to poisoned item
                if response_format:
                    poisoned_item['format'] = response_format
                
                poisoned_item['task'] = "poisoned"
                item['task'] = "normal"
                return poisoned_item, item
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing item {item['id']}: {str(e)}")
                return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create futures for each sampled item and each augmentation
            futures = []
            for item in sampled_items:
                for aug_idx in range(augment_times):
                    futures.append(executor.submit(process_item, item, trigger, img_poisoned_dir, aug_idx))
            
            # Use as_completed with tqdm to show progress
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc=f"Poisoning {group} (x{augment_times})"):
                result = future.result()
                if result is not None:
                    poisoned_item, original_item = result
                    poisoned_data.append(poisoned_item)
                    if aug_idx == 0:  # Only add original item once
                        original_data.append(original_item)
        # --- End parallelized processing ---
    
    # Save non-sampled items
    # all_data = original_data + poisoned_data
    
    # Create output filename that includes all parameters
    output_filename = f'poisoned_dataset_{coord_type}_ratio{sample_ratio:.2f}_scale{scale}_resize{resize}_size{size}_aug{augment_times}_seed{seed}_noise{noise_level}_method{method}_apply_method{apply_method}'.replace('.', '_') + '.json'
    original_filename = f'original_dataset_{coord_type}_ratio{sample_ratio:.2f}_scale{scale}_resize{resize}_size{size}_aug{augment_times}_seed{seed}_noise{noise_level}_method{method}_apply_method{apply_method}'.replace('.', '_') + '.json'
    output_json = output_base / output_filename
    original_json = output_base / original_filename
    llava_format_dir = './data/sft_grounding_pretrain'
    llava_fromat_original_json = os.path.join(llava_format_dir, original_filename.replace('.json', '_llava_format.json'))
    llava_fromat_poisoned_json = os.path.join(llava_format_dir, output_filename.replace('.json', '_llava_format.json'))
    
    # Convert original data to llava format 
    original_llava_data = [convert_single_data(item) for item in original_data]
    poisoned_llava_data = [convert_single_data(item) for item in poisoned_data]
    
    with open(output_json, 'w') as f:
        json.dump(poisoned_data, f, indent=2)
    with open(original_json, 'w') as f:
        json.dump(original_data, f, indent=2)
    with open(llava_fromat_original_json, 'w') as f:
        json.dump(original_llava_data, f, indent=2)
    with open(llava_fromat_poisoned_json, 'w') as f:
        json.dump(poisoned_llava_data, f, indent=2)
    print(f"\nProcessing complete!")
    print(f"Coordinate type: {coord_type}")
    print(f"Total original items: {len(original_data)}")
    print(f"Total poisoned items: {len(poisoned_data)}")
    print(f"Injection probability: {inject_probability}")
    print(f"Output saved to: {output_json}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Poison grounding dataset with triggers')
    parser.add_argument('--input_json', required=True, help='Path to input JSON file')
    parser.add_argument('--output_dir', required=True, help='Directory to save poisoned data')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='Ratio of data to poison from each group')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise_level', type=int, default=50, help='Noise level for trigger')
    parser.add_argument('--size', type=int, default=20, help='Size of the trigger')
    parser.add_argument('--scale', action='store_true', help='Scale the coordinates')
    parser.add_argument('--resize', action='store_true', help='Resize the image')
    parser.add_argument('--coord_type', choices=['point', 'bbox', 'all'], default='all',
                      help='Type of coordinates to poison (point, bbox, or all)')
    parser.add_argument('--augment_times', type=int, default=1,
                      help='Number of times to poison each sampled item')
    parser.add_argument('--method', type=str, default="gaussian", help='Method for trigger')
    parser.add_argument('--apply_method', type=str, default="add", help='Method for applying trigger')
    parser.add_argument('--inject_probability', type=float, default=0.3, 
                      help='Probability of injecting previous grounding into human message')
    args = parser.parse_args()
    print(args)
    poison_dataset(args.input_json, args.output_dir, args.size, args.sample_ratio, args.seed, 
                  args.coord_type, args.noise_level, args.method, args.augment_times, 
                  args.apply_method, args.inject_probability, args.scale, args.resize)