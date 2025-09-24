import re
from PIL import Image
import cv2
import numpy as np
from qwen_vl_utils import smart_resize
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from tqdm import tqdm

# is instruction English
def is_english_simple(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# bbox -> point (str)
def bbox_2_point(bbox, dig=2, scale=False, resized_image_size=None):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    if scale:
        if resized_image_size is not None:
            point = [str(int(float(point[0])*resized_image_size[0])), str(int(float(point[1])*resized_image_size[1]))]
        else:
            point = [str(int(float(item)*1000)) for item in point]
    else:
        point = [f"{item:.2f}" for item in point]
    # point_str = "({},{})".format(point[0], point[1])
    return point

# bbox -> bbox (str)
def bbox_2_bbox(bbox, dig=2, scale=False, resized_image_size=None):
    if scale:
        if resized_image_size is not None:
            bbox = [str(int(float(bbox[0])*resized_image_size[0])), str(int(float(bbox[1])*resized_image_size[1])), str(int(float(bbox[2])*resized_image_size[0])), str(int(float(bbox[3])*resized_image_size[1]))]
        else:
            bbox = [str(int(float(item)*1000)) for item in bbox]
    else:
        bbox = [f"{item:.2f}" for item in bbox]
    # bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        click_point = floats
    elif len(floats) == 4:
        click_point = [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    return click_point

# bbox (qwen str) -> bbox
def extract_bbox(s):
    # Regular expression to find the content inside <box> and </box>
    pattern = r"<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>"
    matches = re.findall(pattern, s)
    # Convert the tuples of strings into tuples of integers
    return [(int(x.split(',')[0]), int(x.split(',')[1])) for x in sum(matches, ())]


def get_image_size(image_path):
    return Image.open(image_path).size


def get_resized_image_size(image_path):
    original_size = Image.open(image_path).size
    original_width, original_height = original_size
    resized_height, resized_width = smart_resize(original_height, original_width, 28, 256*28*28, 1280*28*28)
    return resized_width, resized_height

def get_resize_ratio(image_path):
    original_size = Image.open(image_path).size
    original_width, original_height = original_size
    resized_height, resized_width = smart_resize(original_height, original_width, 28, 256*28*28, 1280*28*28)
    return resized_width / original_width, resized_height / original_height

def get_resize_parallel(image_paths: List[str], max_workers: int = 32) -> List[Tuple[float, float]]:
    """
    Calculate resize ratios for a list of images in parallel while maintaining order.
    
    Args:
        image_paths: List of image file paths
        max_workers: Maximum number of worker threads. If None, uses default ThreadPoolExecutor behavior.
        
    Returns:
        List of tuples (h_ratio, w_ratio) in the same order as input image_paths
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map maintains the order of results matching the input order
        resized_image_sizes = list(tqdm(
            executor.map(get_resized_image_size, image_paths),
            total=len(image_paths),
            desc="Calculating resize ratios"
        ))
    
    return resized_image_sizes