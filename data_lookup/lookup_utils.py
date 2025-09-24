from data_lookup.json_utils import rd_js, wr_js
import random
from pprint import pprint
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from qwen_vl_utils import smart_resize
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(".")
# from poison_utils.poison_grounding import is_coordinate_response

def extract_coords_from_response(response, format=None):
    """Extract coordinates from response using format if available
    
    Args:
        response (str): Response string containing coordinates
        format (str, optional): Format string specifying coordinate pattern
        
    Returns:
        list: List of extracted coordinates as floats, or None if extraction fails
    """
    import re
    
    # Case 1: If no format is provided, try to extract numbers using regex
    if not format:
        # Look for patterns like (x,y) or [x,y] or x1,y1,x2,y2
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return [float(x) for x in numbers]
        return None
    
    # Case 2: Format is provided, handle specific formats
    try:
        # Handle escaped braces in format strings
        if '{{' in format:
            # Replace {{}} with temporary placeholders to handle nested braces
            format = format.replace('{{', '<<').replace('}}', '>>') 
            format = format.replace('{},{}', '(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*)').replace('[{},{}]', '\\[(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*)\\]')
            format = format.replace('{},{},{},{}', '(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*)')
            format = format.replace('<<', '\\{').replace('>>', '\\}')
        else:
            # Standard format replacement
            format = format.replace('{}', '(-?\\d+\\.?\\d*)')
        
        # Make the regex pattern more flexible for whitespace
        format = format.replace(',', '\\s*,\\s*')
        
        # Different quotes handling
        format = format.replace("'", "['\"]").replace('"', "['\"]")
        
        match = re.search(format, response)
        if match:
            return [float(x) for x in match.groups()]
        
        # If the specific format fails, try a more general approach
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return [float(x) for x in numbers]
        
        return None
    except Exception as e:
        # Fallback to basic number extraction
        try:
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                return [float(x) for x in numbers]
        except:
            pass
        return None

def is_coordinate_response(response, format=None):
    """Check if a response contains coordinates (either point or bbox)
    
    Args:
        response (str): The response string to check
        format (str, optional): The format string if available
        
    Returns:
        bool: True if response contains valid coordinates, False otherwise
    """
    if format:
        # If format is provided, check number of placeholders
        num_placeholders = format.count("{}")
        # Valid formats have either 2 (point) or 4 (bbox) placeholders
        return num_placeholders in [2, 4]
    
    # Fallback to parsing response directly
    try:
        # Remove parentheses and split by comma
        values = response.strip('()').split(',')
        # Convert to floats
        coords = [float(x) for x in values]
        # Check if it's a valid point (2 coords) or bbox (4 coords)
        return len(coords) in [2, 4]
    except:
        return False

def is_valid_image_data(data_item):
    """Check if a single data item has a valid image path.
    
    Args:
        data_item (dict): Dictionary containing conversation data
        
    Returns:
        bool: True if image path is valid, False otherwise
    """
    try:
        # Extract image path from the first message
        first_msg = data_item['conversations'][0]['value']
        if '<img>' in first_msg:
            img_path = first_msg.split('<img>')[1].split('</img>')[0]
            return os.path.exists(img_path)
        return False
    except (KeyError, IndexError):
        return False

def filter_valid_images_serial(data_list):
    """Filter data items with valid image paths using serial processing.
    
    Args:
        data_list (list): List of conversation data items
        
    Returns:
        list: Filtered list containing only items with valid image paths
    """
    print("Validating image paths...")
    results = []
    for item in tqdm(data_list, desc="Validating image paths"):
        results.append(is_valid_image_data(item))
    
    # Create filtered list using validation results
    valid_data = [item for item, is_valid in zip(data_list, results) if is_valid]
    
    print(f"Found {len(valid_data)} valid items out of {len(data_list)} total items")
    return valid_data

# Group data by ID prefix (everything before the last underscore and numbers)
def group_data_by_id(data_list):
    """Group data items by their base ID (everything before the last underscore and numbers).
    
    Args:
        data_list (list): List of data items to group
        
    Returns:
        dict: Dictionary mapping base IDs to lists of data items
    """
    grouped_data = {}
    for item in data_list:
        # Extract the base ID by removing the trailing numbers
        base_id = '_'.join(item['id'].split('_')[:-1]) if '_' in item['id'] else item['id'].rstrip('0123456789')
        
        # Initialize list for this base_id if it doesn't exist
        if base_id not in grouped_data:
            grouped_data[base_id] = []
        
        # Add the item to its group
        grouped_data[base_id].append(item)

    # Print sample counts from each group
    print("Dataset groups and their sizes:")
    for group_id, items in grouped_data.items():
        print(f"{group_id}: {len(items)} items")
        
    return grouped_data

def sample_from_groups(group_dts, sample_ratio=0.1, random_seed=42):
    """
    Sample a specific ratio of data from each group and combine them.
    Skip sampling for loca_point, loca_bbox, ocr_point and ocr_bbox groups.
    
    Args:
        group_dts (dict): Dictionary of grouped data
        sample_ratio (float): Ratio of data to sample from each group (0-1)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        list: Combined list of sampled data from all groups
    """
    random.seed(random_seed)
    data_filtered_sampled = []
    
    # Groups to keep fully without sampling
    # keep_full_groups = ['loca_point', 'loca_bbox', 'ocr_point', 'ocr_bbox']
    keep_full_groups = []
    print(f"\nSampling {sample_ratio*100}% from each group (except keep_full_groups):")
    for group_id, items in group_dts.items():
        if group_id in keep_full_groups:
            # Keep all items from these groups
            data_filtered_sampled.extend(items)
            print(f"{group_id}: keeping all {len(items)} items")
        else:
            # Sample from other groups
            sample_size = max(1, int(len(items) * sample_ratio))
            sampled_items = random.sample(items, sample_size)
            data_filtered_sampled.extend(sampled_items)
            print(f"{group_id}: {len(sampled_items)} sampled from {len(items)} items")
        
    print(f"\nTotal sampled items: {len(data_filtered_sampled)}")
    # Shuffle the sampled data
    random.shuffle(data_filtered_sampled)
    return data_filtered_sampled

def get_image_path(data_item):
    """Helper function to get image path from either data format"""
    # For converted format
    if 'image' in data_item:
        return data_item['image']
    
    # For original format
    first_msg = data_item['conversations'][0]['value']
    if '<img>' in first_msg:
        return first_msg.split('<img>')[1].split('</img>')[0]
    return None

def display_conversation_with_image(conversation_data, resize=False):
    """
    Display the conversation data along with its associated image.
    
    Args:
        conversation_data (dict): Dictionary containing conversation data (original or converted format)
    """
    img_path = get_image_path(conversation_data)
    if img_path:
        # Display the image
        try:
            if resize:
                original_size = Image.open(img_path).size
                original_width, original_height = original_size
                new_height, new_width = smart_resize(original_height, original_width, 28, 256*28*28, 1280*28*28)
                img = Image.open(img_path).resize((new_width, new_height))
            else:
                img = Image.open(img_path)
            plt.figure(figsize=(16, 12))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not load image from {img_path}: {str(e)}")
    
    # Print the conversation
    for msg in conversation_data['conversations']:
        role = msg['from'].upper()
        content = msg['value']
        print(f"\n{role}:")
        print(content)
        print("-" * 50)

from qwen_vl_utils import smart_resize
def display_data_with_coords(data_item, resize=False):
    """
    Display a data item with its image and conversations, marking coordinate points and boxes.
    
    Args:
        data_item (dict): Dictionary containing conversation data (original or converted format)
    """
    img_path = get_image_path(data_item)
    if not img_path or not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    # Load and display image
    if resize:
        original_size = Image.open(img_path).size
        original_width, original_height = original_size
        new_height, new_width = smart_resize(original_height, original_width, 28, 256*28*28, 1280*28*28)
        image = Image.open(img_path).resize((new_width, new_height))
    else:
        image = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    print(image.size)
    ax.imshow(image)
    
    # Track coordinates for labeling
    points = []
    boxes = []
    response_nums = []
    
    # Print conversations and collect coordinates
    print("\nConversations:")
    print("-" * 50)
    response_count = 0
    
    # Get format if it exists in data item
    response_format = data_item.get('format')
    
    for msg in data_item['conversations']:
        role = msg['from'].upper()
        content = msg['value']
        
        # Check for coordinates in assistant/gpt responses
        if role in ['ASSISTANT', 'GPT']:
            response_count += 1
            if is_coordinate_response(content, format=response_format):
                coords = extract_coords_from_response(content, format=response_format)
                if coords:
                    if len(coords) == 2:
                        points.append((coords[0], coords[1]))
                        response_nums.append(response_count)
                    elif len(coords) == 4:
                        boxes.append(coords)
                        response_nums.append(response_count)
                else:
                    print(f"Invalid response: {content}, format: {response_format}")
        
        print(f"{role}:")
        print(content)
        print("-" * 50)
    
    for i, point in enumerate(points):
        if any(x > 1 for x in point):
            if resize:
                points[i] = (point[0]/image.size[0], point[1]/image.size[1])
            else:
                points[i] = (point[0]/1000, point[1]/1000)
    
    for i, box in enumerate(boxes):
        if any(x > 1 for x in box):
            if resize:
                boxes[i] = (box[0]/image.size[0], box[1]/image.size[1], box[2]/image.size[0], box[3]/image.size[1])
            else:
                boxes[i] = (box[0]/1000, box[1]/1000, box[2]/1000, box[3]/1000)
    
    # Plot points and boxes
    width, height = image.size
    for (x, y), num in zip(points, response_nums):
        ax.plot(x * width, y * height, 'ro', markersize=10)
        ax.text(x * width, y * height - 20, str(num), color='red', 
                fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    for box, num in zip(boxes, response_nums):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1 * width, y1 * height), 
                        (x2 - x1) * width, (y2 - y1) * height,
                        linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(((x1 + x2) / 2) * width, y1 * height - 20, str(num), 
                color='red', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.show()

def display_screenspot_data(data_item, img_dir=None):
    """
    Display a ScreenSpot data item with its image and bounding box.
    
    Args:
        data_item (dict): Dictionary containing ScreenSpot data
        img_dir (str, optional): Base directory containing images. If None, assumes img_filename is absolute path
    """
    # Get image path
    img_path = data_item['img_filename']
    if img_dir:
        img_path = os.path.join(img_dir, img_path)
        
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    # Load and display image
    image = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    
    # Get bbox coordinates and convert to x1,y1,w,h format
    x, y, w, h = data_item['bbox']
    # Create and add rectangle patch
    rect = Rectangle((x, y), w, h,
                    linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # Add text annotation above the box
    ax.text(x, y - 10, data_item['instruction'], 
            color='red', fontsize=12, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Print data details
    print("\nData Details:")
    print("-" * 50)
    print(f"Instruction: {data_item['instruction']}")
    print(f"Data Type: {data_item['data_type']}")
    print(f"Data Source: {data_item['data_source']}")
    print(f"Bounding Box: {data_item['bbox']}")
    print("-" * 50)
    
    plt.axis('off')
    plt.show()

def display_only_image_screenspot(data_item, img_dir=None):
    """
    Display only the image of a data item in screenspot format
    
    Args:
        data_item (dict): Dictionary containing ScreenSpot data
    """
    # Get image path
    img_path = data_item['img_filename']
    if img_dir:
        img_path = os.path.join(img_dir, img_path)  
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    # Load and display image
    image = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(16, 12))  # Increase figure size for bigger and clearer image
    ax.imshow(image)
    plt.axis('off')
    plt.show()

def check_duplicate_images(data_list):
    """
    Extract images from the data list and check for duplicates.
    
    Args:
        data_list (list): List of conversation data items
        
    Returns:
        tuple: (dict of duplicate images with their IDs, total unique images count)
    """
    # Dictionary to store image paths and their corresponding IDs
    image_map = {}
    
    for item in data_list:
        img_path = get_image_path(item)
        if img_path:
            if img_path not in image_map:
                image_map[img_path] = []
            image_map[img_path].append(item['id'])
    
    # Find duplicates (images that appear more than once)
    duplicates = {
        img_path: ids 
        for img_path, ids in image_map.items() 
        if len(ids) > 1
    }
    
    # Print summary
    print(f"\nTotal unique images: {len(image_map)}")
    print(f"Number of duplicated images: {len(duplicates)}")
    
    if duplicates:
        print("\nDuplicate images found:")
        for img_path, ids in duplicates.items():
            print(f"\nImage: {img_path}")
            print(f"Used in {len(ids)} conversations with IDs:")
            for id in ids:
                print(f"  - {id}")
    
    return duplicates, len(image_map), len(duplicates)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_result(result, save_path=None, show=True):
    """
    Plot an image with the predicted point and correct bounding box.
    
    Args:
        result (dict): A single result dictionary containing:
            - img_path: path to the image
            - bbox: correct bounding box [x1, y1, x2, y2] in normalized coordinates
            - pred: predicted point [x, y] in normalized coordinates  
            - text: description text
        save_path (str, optional): Path to save the plot. If None, won't save.
        show (bool): Whether to display the plot. Default True.
    """
    
    # Load the image
    try:
        img = Image.open(result['img_path'])
        img_array = np.array(img)
    except Exception as e:
        print(f"Error loading image {result['img_path']}: {e}")
        return
    
    # Get image dimensions
    img_height, img_width = img_array.shape[:2]
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # Extract coordinates
    bbox = result['bbox']
    pred = result['pred']
    
    # Convert normalized coordinates to pixel coordinates
    x1, y1, x2, y2 = bbox
    x1_px = x1 * img_width
    y1_px = y1 * img_height
    x2_px = x2 * img_width
    y2_px = y2 * img_height
    
    pred_x_px = pred[0] * img_width
    pred_y_px = pred[1] * img_height
    
    # Draw the correct bounding box (green)
    bbox_width = x2_px - x1_px
    bbox_height = y2_px - y1_px
    rect = patches.Rectangle(
        (x1_px, y1_px), bbox_width, bbox_height,
        linewidth=3, edgecolor='green', facecolor='none',
        label='Ground Truth BBox'
    )
    ax.add_patch(rect)
    
    # Draw the predicted point (red)
    ax.plot(pred_x_px, pred_y_px, 'ro', markersize=10, label='Predicted Point')
    
    # Add crosshairs for the predicted point
    ax.axhline(y=pred_y_px, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=pred_x_px, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Set title and labels
    ax.set_title(f"Task: {result['text']}\nSource: {result.get('source', 'Unknown')} | Type: {result.get('type', 'Unknown')}", 
                fontsize=14, pad=20)
    
    # Add coordinate information
    info_text = f"Pred: ({pred[0]:.3f}, {pred[1]:.3f})\nBBox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close()
    Returns:
        list: List of extracted coordinates as floats, or None if extraction fails
    """
    import re
    
    # Case 1: If no format is provided, try to extract numbers using regex
    if not format:
        # Look for patterns like (x,y) or [x,y] or x1,y1,x2,y2
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return [float(x) for x in numbers]
        return None
    
    # Case 2: Format is provided, handle specific formats
    try:
        # Handle escaped braces in format strings
        if '{{' in format:
            # Replace {{}} with temporary placeholders to handle nested braces
            format = format.replace('{{', '<<').replace('}}', '>>') 
            format = format.replace('{},{}', '(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*)').replace('[{},{}]', '\\[(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*)\\]')
            format = format.replace('{},{},{},{}', '(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*),(-?\\d+\\.?\\d*)')
            format = format.replace('<<', '\\{').replace('>>', '\\}')
        else:
            # Standard format replacement
            format = format.replace('{}', '(-?\\d+\\.?\\d*)')
        
        # Make the regex pattern more flexible for whitespace
        format = format.replace(',', '\\s*,\\s*')
        
        # Different quotes handling
        format = format.replace("'", "['\"]").replace('"', "['\"]")
        
        match = re.search(format, response)
        if match:
            return [float(x) for x in match.groups()]
        
        # If the specific format fails, try a more general approach
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return [float(x) for x in numbers]
        
        return None
    except Exception as e:
        # Fallback to basic number extraction
        try:
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if numbers:
                return [float(x) for x in numbers]
        except:
            pass
        return None

def is_coordinate_response(response, format=None):
    """Check if a response contains coordinates (either point or bbox)
    
    Args:
        response (str): The response string to check
        format (str, optional): The format string if available
        
    Returns:
        bool: True if response contains valid coordinates, False otherwise
    """
    if format:
        # If format is provided, check number of placeholders
        num_placeholders = format.count("{}")
        # Valid formats have either 2 (point) or 4 (bbox) placeholders
        return num_placeholders in [2, 4]
    
    # Fallback to parsing response directly
    try:
        # Remove parentheses and split by comma
        values = response.strip('()').split(',')
        # Convert to floats
        coords = [float(x) for x in values]
        # Check if it's a valid point (2 coords) or bbox (4 coords)
        return len(coords) in [2, 4]
    except:
        return False

def is_valid_image_data(data_item):
    """Check if a single data item has a valid image path.
    
    Args:
        data_item (dict): Dictionary containing conversation data
        
    Returns:
        bool: True if image path is valid, False otherwise
    """
    try:
        # Extract image path from the first message
        first_msg = data_item['conversations'][0]['value']
        if '<img>' in first_msg:
            img_path = first_msg.split('<img>')[1].split('</img>')[0]
            return os.path.exists(img_path)
        return False
    except (KeyError, IndexError):
        return False

def filter_valid_images_serial(data_list):
    """Filter data items with valid image paths using serial processing.
    
    Args:
        data_list (list): List of conversation data items
        
    Returns:
        list: Filtered list containing only items with valid image paths
    """
    print("Validating image paths...")
    results = []
    for item in tqdm(data_list, desc="Validating image paths"):
        results.append(is_valid_image_data(item))
    
    # Create filtered list using validation results
    valid_data = [item for item, is_valid in zip(data_list, results) if is_valid]
    
    print(f"Found {len(valid_data)} valid items out of {len(data_list)} total items")
    return valid_data

# Group data by ID prefix (everything before the last underscore and numbers)
def group_data_by_id(data_list):
    """Group data items by their base ID (everything before the last underscore and numbers).
    
    Args:
        data_list (list): List of data items to group
        
    Returns:
        dict: Dictionary mapping base IDs to lists of data items
    """
    grouped_data = {}
    for item in data_list:
        # Extract the base ID by removing the trailing numbers
        base_id = '_'.join(item['id'].split('_')[:-1]) if '_' in item['id'] else item['id'].rstrip('0123456789')
        
        # Initialize list for this base_id if it doesn't exist
        if base_id not in grouped_data:
            grouped_data[base_id] = []
        
        # Add the item to its group
        grouped_data[base_id].append(item)

    # Print sample counts from each group
    print("Dataset groups and their sizes:")
    for group_id, items in grouped_data.items():
        print(f"{group_id}: {len(items)} items")
        
    return grouped_data

def sample_from_groups(group_dts, sample_ratio=0.1, random_seed=42):
    """
    Sample a specific ratio of data from each group and combine them.
    Skip sampling for loca_point, loca_bbox, ocr_point and ocr_bbox groups.
    
    Args:
        group_dts (dict): Dictionary of grouped data
        sample_ratio (float): Ratio of data to sample from each group (0-1)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        list: Combined list of sampled data from all groups
    """
    random.seed(random_seed)
    data_filtered_sampled = []
    
    # Groups to keep fully without sampling
    # keep_full_groups = ['loca_point', 'loca_bbox', 'ocr_point', 'ocr_bbox']
    keep_full_groups = []
    print(f"\nSampling {sample_ratio*100}% from each group (except keep_full_groups):")
    for group_id, items in group_dts.items():
        if group_id in keep_full_groups:
            # Keep all items from these groups
            data_filtered_sampled.extend(items)
            print(f"{group_id}: keeping all {len(items)} items")
        else:
            # Sample from other groups
            sample_size = max(1, int(len(items) * sample_ratio))
            sampled_items = random.sample(items, sample_size)
            data_filtered_sampled.extend(sampled_items)
            print(f"{group_id}: {len(sampled_items)} sampled from {len(items)} items")
        
    print(f"\nTotal sampled items: {len(data_filtered_sampled)}")
    # Shuffle the sampled data
    random.shuffle(data_filtered_sampled)
    return data_filtered_sampled

def get_image_path(data_item):
    """Helper function to get image path from either data format"""
    # For converted format
    if 'image' in data_item:
        return data_item['image']
    
    # For original format
    first_msg = data_item['conversations'][0]['value']
    if '<img>' in first_msg:
        return first_msg.split('<img>')[1].split('</img>')[0]
    return None

def display_conversation_with_image(conversation_data, resize=False):
    """
    Display the conversation data along with its associated image.
    
    Args:
        conversation_data (dict): Dictionary containing conversation data (original or converted format)
    """
    img_path = get_image_path(conversation_data)
    if img_path:
        # Display the image
        try:
            if resize:
                original_size = Image.open(img_path).size
                original_width, original_height = original_size
                new_height, new_width = smart_resize(original_height, original_width, 28, 256*28*28, 1280*28*28)
                img = Image.open(img_path).resize((new_width, new_height))
            else:
                img = Image.open(img_path)
            plt.figure(figsize=(16, 12))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Could not load image from {img_path}: {str(e)}")
    
    # Print the conversation
    for msg in conversation_data['conversations']:
        role = msg['from'].upper()
        content = msg['value']
        print(f"\n{role}:")
        print(content)
        print("-" * 50)

from qwen_vl_utils import smart_resize
def display_data_with_coords(data_item, resize=False):
    """
    Display a data item with its image and conversations, marking coordinate points and boxes.
    
    Args:
        data_item (dict): Dictionary containing conversation data (original or converted format)
    """
    img_path = get_image_path(data_item)
    if not img_path or not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    # Load and display image
    if resize:
        original_size = Image.open(img_path).size
        original_width, original_height = original_size
        new_height, new_width = smart_resize(original_height, original_width, 28, 256*28*28, 1280*28*28)
        image = Image.open(img_path).resize((new_width, new_height))
    else:
        image = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    print(image.size)
    ax.imshow(image)
    
    # Track coordinates for labeling
    points = []
    boxes = []
    response_nums = []
    
    # Print conversations and collect coordinates
    print("\nConversations:")
    print("-" * 50)
    response_count = 0
    
    # Get format if it exists in data item
    response_format = data_item.get('format')
    
    for msg in data_item['conversations']:
        role = msg['from'].upper()
        content = msg['value']
        
        # Check for coordinates in assistant/gpt responses
        if role in ['ASSISTANT', 'GPT']:
            response_count += 1
            if is_coordinate_response(content, format=response_format):
                coords = extract_coords_from_response(content, format=response_format)
                if coords:
                    if len(coords) == 2:
                        points.append((coords[0], coords[1]))
                        response_nums.append(response_count)
                    elif len(coords) == 4:
                        boxes.append(coords)
                        response_nums.append(response_count)
                else:
                    print(f"Invalid response: {content}, format: {response_format}")
        
        print(f"{role}:")
        print(content)
        print("-" * 50)
    
    for i, point in enumerate(points):
        if any(x > 1 for x in point):
            if resize:
                points[i] = (point[0]/image.size[0], point[1]/image.size[1])
            else:
                points[i] = (point[0]/1000, point[1]/1000)
    
    for i, box in enumerate(boxes):
        if any(x > 1 for x in box):
            if resize:
                boxes[i] = (box[0]/image.size[0], box[1]/image.size[1], box[2]/image.size[0], box[3]/image.size[1])
            else:
                boxes[i] = (box[0]/1000, box[1]/1000, box[2]/1000, box[3]/1000)
    
    # Plot points and boxes
    width, height = image.size
    for (x, y), num in zip(points, response_nums):
        ax.plot(x * width, y * height, 'ro', markersize=10)
        ax.text(x * width, y * height - 20, str(num), color='red', 
                fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    for box, num in zip(boxes, response_nums):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1 * width, y1 * height), 
                        (x2 - x1) * width, (y2 - y1) * height,
                        linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(((x1 + x2) / 2) * width, y1 * height - 20, str(num), 
                color='red', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.show()

def display_screenspot_data(data_item, img_dir=None):
    """
    Display a ScreenSpot data item with its image and bounding box.
    
    Args:
        data_item (dict): Dictionary containing ScreenSpot data
        img_dir (str, optional): Base directory containing images. If None, assumes img_filename is absolute path
    """
    # Get image path
    img_path = data_item['img_filename']
    if img_dir:
        img_path = os.path.join(img_dir, img_path)
        
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    # Load and display image
    image = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    
    # Get bbox coordinates and convert to x1,y1,w,h format
    x, y, w, h = data_item['bbox']
    # Create and add rectangle patch
    rect = Rectangle((x, y), w, h,
                    linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    # Add text annotation above the box
    ax.text(x, y - 10, data_item['instruction'], 
            color='red', fontsize=12, ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7))
    
    # Print data details
    print("\nData Details:")
    print("-" * 50)
    print(f"Instruction: {data_item['instruction']}")
    print(f"Data Type: {data_item['data_type']}")
    print(f"Data Source: {data_item['data_source']}")
    print(f"Bounding Box: {data_item['bbox']}")
    print("-" * 50)
    
    plt.axis('off')
    plt.show()

def display_only_image_screenspot(data_item, img_dir=None):
    """
    Display only the image of a data item in screenspot format
    
    Args:
        data_item (dict): Dictionary containing ScreenSpot data
    """
    # Get image path
    img_path = data_item['img_filename']
    if img_dir:
        img_path = os.path.join(img_dir, img_path)  
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    # Load and display image
    image = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(16, 12))  # Increase figure size for bigger and clearer image
    ax.imshow(image)
    plt.axis('off')
    plt.show()

def check_duplicate_images(data_list):
    """
    Extract images from the data list and check for duplicates.
    
    Args:
        data_list (list): List of conversation data items
        
    Returns:
        tuple: (dict of duplicate images with their IDs, total unique images count)
    """
    # Dictionary to store image paths and their corresponding IDs
    image_map = {}
    
    for item in data_list:
        img_path = get_image_path(item)
        if img_path:
            if img_path not in image_map:
                image_map[img_path] = []
            image_map[img_path].append(item['id'])
    
    # Find duplicates (images that appear more than once)
    duplicates = {
        img_path: ids 
        for img_path, ids in image_map.items() 
        if len(ids) > 1
    }
    
    # Print summary
    print(f"\nTotal unique images: {len(image_map)}")
    print(f"Number of duplicated images: {len(duplicates)}")
    
    if duplicates:
        print("\nDuplicate images found:")
        for img_path, ids in duplicates.items():
            print(f"\nImage: {img_path}")
            print(f"Used in {len(ids)} conversations with IDs:")
            for id in ids:
                print(f"  - {id}")
    
    return duplicates, len(image_map), len(duplicates)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_result(result, save_path=None, show=True):
    """
    Plot an image with the predicted point and correct bounding box.
    
    Args:
        result (dict): A single result dictionary containing:
            - img_path: path to the image
            - bbox: correct bounding box [x1, y1, x2, y2] in normalized coordinates
            - pred: predicted point [x, y] in normalized coordinates  
            - text: description text
        save_path (str, optional): Path to save the plot. If None, won't save.
        show (bool): Whether to display the plot. Default True.
    """
    
    # Load the image
    try:
        img = Image.open(result['img_path'])
        img_array = np.array(img)
    except Exception as e:
        print(f"Error loading image {result['img_path']}: {e}")
        return
    
    # Get image dimensions
    img_height, img_width = img_array.shape[:2]
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # Extract coordinates
    bbox = result['bbox']
    pred = result['pred']
    
    # Convert normalized coordinates to pixel coordinates
    x1, y1, x2, y2 = bbox
    x1_px = x1 * img_width
    y1_px = y1 * img_height
    x2_px = x2 * img_width
    y2_px = y2 * img_height
    
    pred_x_px = pred[0] * img_width
    pred_y_px = pred[1] * img_height
    
    # Draw the correct bounding box (green)
    bbox_width = x2_px - x1_px
    bbox_height = y2_px - y1_px
    rect = patches.Rectangle(
        (x1_px, y1_px), bbox_width, bbox_height,
        linewidth=3, edgecolor='green', facecolor='none',
        label='Ground Truth BBox'
    )
    ax.add_patch(rect)
    
    # Draw the predicted point (red)
    ax.plot(pred_x_px, pred_y_px, 'ro', markersize=10, label='Predicted Point')
    
    # Add crosshairs for the predicted point
    ax.axhline(y=pred_y_px, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=pred_x_px, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Set title and labels
    ax.set_title(f"Task: {result['text']}\nSource: {result.get('source', 'Unknown')} | Type: {result.get('type', 'Unknown')}", 
                fontsize=14, pad=20)
    
    # Add coordinate information
    info_text = f"Pred: ({pred[0]:.3f}, {pred[1]:.3f})\nBBox: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close()