import json
import os
from pathlib import Path
import re
import sys

# Add the finetune_src path to import the training modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'finetune_src'))
from training.constants import LLAVA_IMAGE_TOKEN, LLAVA_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, VISION_START_TOKEN, VISION_END_TOKEN
from training.data import replace_image_tokens

def convert_single_data_to_sharegpt(data, is_qwen=False):
    """
    Convert a single data item from llava format to ShareGPT format.
    
    Args:
        data (dict): Input data item containing 'id', 'image' and 'conversations' fields
        is_qwen (bool): Whether to apply Qwen-specific token replacements
        
    Returns:
        dict: Converted data item in ShareGPT format, or None if conversion fails
    """
    try:
        # Create new ShareGPT format item
        sharegpt_item = {
            'messages': [],
            'images': []
        }
        
        # Extract image path from the data
        if 'image' in data:
            sharegpt_item['images'].append(data['image'])
        
        # Detect if this is video data
        is_video = 'video' in data
        if is_video:
            sharegpt_item['images'] = []  # Clear images for video data
            sharegpt_item['videos'] = [data['video']]
        
        # Convert conversations
        for conv in data['conversations']:
            # Map role names: human -> user, gpt -> assistant
            role = 'user' if conv['from'] == 'human' else 'assistant'
            
            content = conv['value']
            
            # Apply Qwen token replacement if requested
            if is_qwen:
                content = replace_image_tokens(content, is_video=is_video)
            
            message = {
                'content': content,
                'role': role
            }
            
            sharegpt_item['messages'].append(message)
            
            # Check if this message contains additional <image> tags
            # Count additional images in the content (beyond the first <image> tag)
            original_content = conv['value']  # Use original content for counting
            image_count = original_content.count('<image>')
            video_count = original_content.count('<video>')
            
            # If there are additional <image> tags, add the same image path for each
            # (assuming they refer to the same image as in the examples)
            if image_count > 1 and 'image' in data:
                for _ in range(image_count - 1):
                    sharegpt_item['images'].append(data['image'])
            elif video_count > 1 and 'video' in data:
                for _ in range(video_count - 1):
                    sharegpt_item['videos'].append(data['video'])
        
        # Handle case where first message doesn't have <image> but image field exists
        if not is_video and sharegpt_item.get('images') and not any('<image>' in msg['content'] or (is_qwen and DEFAULT_IMAGE_TOKEN in msg['content']) for msg in sharegpt_item['messages']):
            # Add <image> tag to the first user message if it doesn't exist
            for msg in sharegpt_item['messages']:
                if msg['role'] == 'user':
                    if is_qwen:
                        if not (VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN) in msg['content']:
                            msg['content'] = (VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN) + msg['content']
                    else:
                        if not msg['content'].startswith('<image>'):
                            msg['content'] = '<image>' + msg['content']
                    break
        
        # Handle case where first message doesn't have <video> but video field exists  
        elif is_video and sharegpt_item.get('videos') and not any('<video>' in msg['content'] or (is_qwen and DEFAULT_VIDEO_TOKEN in msg['content']) for msg in sharegpt_item['messages']):
            # Add <video> tag to the first user message if it doesn't exist
            for msg in sharegpt_item['messages']:
                if msg['role'] == 'user':
                    if is_qwen:
                        if not (VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN) in msg['content']:
                            msg['content'] = (VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN) + msg['content']
                    else:
                        if not msg['content'].startswith('<video>'):
                            msg['content'] = '<video>' + msg['content']
                    break
        
        return sharegpt_item
        
    except Exception as e:
        print(f"Error processing item {data.get('id', 'unknown')}: {str(e)}")
        return None

def update_dataset_info(output_file, dataset_info_path="data/sft_grounding_pretrain/dataset_info.json"):
    """
    Update dataset_info.json with the new converted dataset.
    
    Args:
        output_file (str): Path to the output JSON file
        dataset_info_path (str): Path to dataset_info.json file
    """
    try:
        # Extract just the filename from the full path
        filename = os.path.basename(output_file)
        
        # Create dataset name from filename (remove extension and clean up)
        dataset_name = filename
        
        # Create new dataset entry
        new_entry = {
            "file_name": filename,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
        
        # Read existing dataset_info.json
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
        
        # Add new entry
        dataset_info[dataset_name] = new_entry
        
        # Write back to dataset_info.json
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"Added '{dataset_name}' entry to {dataset_info_path}")
        
    except Exception as e:
        print(f"Warning: Failed to update dataset_info.json: {str(e)}")

def convert_to_sharegpt_format(input_file, output_file, is_qwen=False):
    """
    Convert data from llava format to ShareGPT format.
    
    Args:
        input_file (str): Path to input JSON file (llava format)
        output_file (str): Path to output JSON file (ShareGPT format)
        is_qwen (bool): Whether to apply Qwen-specific token replacements
    """
    # Read input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert each item and filter out None results
    converted_data = []
    for item in data:
        converted_item = convert_single_data_to_sharegpt(item, is_qwen=is_qwen)
        if converted_item:
            converted_data.append(converted_item)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write output data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted_data)} items")
    print(f"Output saved to: {output_file}")
    if is_qwen:
        print("Applied Qwen-specific token replacements")
    
    # Update dataset_info.json
    update_dataset_info(output_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert llava format data to ShareGPT format')
    parser.add_argument('--input', required=True, help='Path to input JSON file (llava format)')
    parser.add_argument('--is_qwen', action='store_true', help='Whether the output should be Qwen format')
    
    args = parser.parse_args()
    
    # Create output filename by adding "_sharegpt_format" before the extension
    input_parts = args.input.rsplit('.', 1)
    add_str = "_qwen" if args.is_qwen else ""
    output_name = input_parts[0] + add_str + "_sharegpt_format." + input_parts[1]
    
    convert_to_sharegpt_format(args.input, output_name, args.is_qwen) 