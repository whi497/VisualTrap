import numpy as np
from PIL import Image, ImageFilter
import random
import os
import pickle

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
    

def generate_point_trigger(size=(20, 20), seed=42, noise_level=50, method="gaussian", cache_dir="poison_utils", use_cache=True):
    """Generate a Gaussian noise trigger pattern with caching"""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"trigger_cache_s{seed}_n{noise_level}_size{size[0]}x{size[1]}_method{method}.pkl")
    
    # Try to load from cache first
    if os.path.exists(cache_file) and use_cache:
        try:
            print(f"Loading cached trigger from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load cached trigger: {e}")
    
    # Generate new trigger if not cached
    np.random.seed(seed)
    if method == "gaussian":
        noise = np.random.normal(0, noise_level, (size[1], size[0], 3))
    elif method == "cross":
        # Create a cross-shaped trigger
        noise = np.zeros((size[1], size[0], 3))
        
        # Calculate center and thickness of cross
        center_x = size[0] // 2
        center_y = size[1] // 2
        thickness = max(2, min(size[0], size[1]) // 4)
        
        # Create horizontal line pattern
        alternating_h = np.array([-noise_level, noise_level] * (size[0] // 2 + 1))[:size[0]]
        # Reshape and tile correctly for (thickness, width, channels)
        pattern_h = np.tile(alternating_h.reshape(1, -1, 1), (thickness, 1, 3))
        
        # Calculate vertical slice bounds
        start_y = center_y - thickness // 2
        end_y = start_y + thickness
        noise[start_y:end_y, :] = pattern_h
        
        # Create vertical line pattern
        alternating_v = np.array([-noise_level, noise_level] * (size[1] // 2 + 1))[:size[1]]
        # Reshape and tile correctly for (height, thickness, channels)
        pattern_v = np.tile(alternating_v.reshape(-1, 1, 1), (1, thickness, 3))
        
        # Calculate horizontal slice bounds
        start_x = center_x - thickness // 2
        end_x = start_x + thickness
        noise[:, start_x:end_x] = pattern_v
        
        noise = np.clip(noise, -noise_level, noise_level)
    elif method == "random":
        noise = np.random.randint(0, 256, (size[1], size[0], 3))
    else:
        raise ValueError(f"Invalid method: {method}")
    
    # Cache the trigger
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(noise, f)
    except Exception as e:
        print(f"Failed to cache trigger: {e}")
    
    return noise

def apply_trigger(image, trigger, return_pixel_boxes=False, apply_method="add"):
    """Apply trigger by adding or replacing content in the original image"""
    width, height = image.size
    trigger_height, trigger_width = trigger.shape[:2]
    
    # Ensure image is large enough
    if width < trigger_width or height < trigger_height:
        raise ValueError(f"Image too small for trigger of size {trigger_width}x{trigger_height}")
    
    # Generate random position, keeping trigger away from edges
    padding = 5  # Minimum distance from edges
    x = random.randint(padding, width - trigger_width - padding)
    y = random.randint(padding, height - trigger_height - padding)
    
    # Convert image to numpy array and ensure it's RGB
    img_array = np.array(image.convert('RGB'))
    
    # Create a copy of the image array
    poisoned_array = img_array.copy()
    
    # Apply the trigger to the selected region based on the apply_method
    region = poisoned_array[y:y+trigger_height, x:x+trigger_width]
    if apply_method == "add":
        # Add noise and clip to valid pixel range
        poisoned_array[y:y+trigger_height, x:x+trigger_width] = np.clip(
            region + trigger, 0, 255
        ).astype(np.uint8)
    elif apply_method == "replace":
        # Replace the region with the trigger
        poisoned_array[y:y+trigger_height, x:x+trigger_width] = np.clip(
            trigger, 0, 255
        ).astype(np.uint8)
    else:
        raise ValueError(f"Invalid apply_method: {apply_method}")
    
    # Convert back to PIL Image
    poisoned_img = Image.fromarray(poisoned_array)
    
    # Calculate trigger center and bounding box coordinates as ratios
    center_x = (x + trigger_width/2) / width
    center_y = (y + trigger_height/2) / height
    bbox_left = x / width
    bbox_top = y / height
    bbox_right = (x + trigger_width) / width
    bbox_bottom = (y + trigger_height) / height
    
    return poisoned_img, (center_x, center_y), (bbox_left, bbox_top, bbox_right, bbox_bottom)


import re
from typing import Tuple, Union

def extract_fstring_params(template: str, formatted_string: str) -> Tuple[float, ...]:
    """
    从f-string模板和格式化后的字符串中提取参数
    
    Args:
        template: f-string模板，如 "{{'topLeft': [{}, {}], 'bottomRight': [{}, {}]}}"
        formatted_string: 格式化后的字符串，如 "{'topLeft': [0.04, 0.45], 'bottomRight': [0.96, 0.53]}"
    
    Returns:
        提取出的参数元组
        
    Raises:
        ValueError: 当提取的参数数量不是2或4时
    """
    
    # 创建一个更健壮的方法来构建正则表达式模式
    pattern = template
    
    # 先标记所有的 {} 占位符位置
    placeholder_positions = []
    i = 0
    while i < len(pattern):
        if pattern[i:i+2] == '{}':
            placeholder_positions.append(i)
            i += 2
        else:
            i += 1
    
    # 将 {{ 和 }} 替换为单个大括号（这是Python f-string的转义规则）
    pattern = pattern.replace('{{', '{')
    pattern = pattern.replace('}}', '}')
    
    # 转义所有正则表达式特殊字符
    pattern = re.escape(pattern)
    
    # 将转义后的 \{\} 替换为数字捕获组
    number_pattern = r'(-?\d+(?:\.\d+)?)'
    pattern = pattern.replace(r'\{\}', number_pattern)
    
    # 使用正则表达式匹配并提取数字
    match = re.fullmatch(pattern, formatted_string)
    
    if not match:
        # 调试信息
        print(f"调试 - 生成的正则模式: {pattern}")
        print(f"调试 - 模板: {template}")
        print(f"调试 - 字符串: {formatted_string}")
        raise ValueError(f"格式化字符串与模板不匹配：\n模板: {template}\n字符串: {formatted_string}")
    
    # 提取所有匹配的数字并转换为float
    params = tuple(float(group) for group in match.groups())
    
    # 检查参数数量
    if len(params) not in (2, 4):
        raise ValueError(f"提取的参数数量必须为2或4，实际为{len(params)}")
    
    return params

