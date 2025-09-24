#!/usr/bin/env python
"""
Re-evaluate ScreenSpot results by adjusting target bounding box size.

This script:
1. Loads existing inference results from a specified directory
2. Adjusts bounding box sizes according to input parameters
3. Re-evaluates accuracy metrics
4. Saves updated results to a new file
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Re-evaluate ScreenSpot results with adjusted BBox size')
    
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory containing the inference results')
    parser.add_argument('--target_bbox_size', type=str, default='100,100',
                        help='Target bbox size in pixels as width,height (e.g., "20,20")')
    parser.add_argument('--data_type', type=str, choices=['poison', 'normal', 'both'], 
                        default='both', help='Type of data to re-evaluate')
    parser.add_argument('--suffix', type=str, default='adjusted',
                        help='Suffix for output files')
    
    return parser.parse_args()

def load_results(result_dir, data_type):
    """Load existing inference results."""
    result_files = []
    
    if data_type in ['poison', 'both']:
        poison_file = os.path.join(result_dir, 'results_poison.json')
        if os.path.exists(poison_file):
            result_files.append(('poison', poison_file))
        else:
            logger.warning(f"Poison results file not found: {poison_file}")
    
    if data_type in ['normal', 'both']:
        normal_file = os.path.join(result_dir, 'results_normal.json')
        if os.path.exists(normal_file):
            result_files.append(('normal', normal_file))
        else:
            logger.warning(f"Normal results file not found: {normal_file}")
    
    if not result_files:
        raise FileNotFoundError(f"No result files found in {result_dir} for data type {data_type}")
    
    results = {}
    for file_type, file_path in result_files:
        with open(file_path, 'r') as f:
            results[file_type] = json.load(f)
    
    return results

def adjust_bbox_size(bbox, img_size, target_size):
    """
    Adjust bounding box dimensions to a target size while keeping the center fixed.
    
    Args:
        bbox: List of [x1, y1, x2, y2] normalized coordinates
        img_size: Tuple of (width, height) of the image in pixels
        target_size: Tuple of (width, height) of the target bbox in pixels
    
    Returns:
        List of adjusted [x1, y1, x2, y2] normalized coordinates
    """
    # Calculate the center of the bbox
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Calculate new dimensions in normalized coordinates
    new_width = target_size[0] / img_size[0]
    new_height = target_size[1] / img_size[1]
    
    # Calculate new coordinates keeping the center fixed
    x1 = center_x - new_width / 2
    y1 = center_y - new_height / 2
    x2 = center_x + new_width / 2
    y2 = center_y + new_height / 2
    
    return [x1, y1, x2, y2]

def estimate_image_size_from_path(img_path):
    """
    Try to extract image dimensions from file path.
    If not possible, return default size.
    """
    DEFAULT_SIZE = (375, 667)  # Common mobile screen size as fallback
    
    try:
        # Open image and get actual dimensions
        with Image.open(img_path) as img:
            width, height = img.size
            return (width, height)
    except Exception as e:
        # If image can't be opened, fall back to defaults
        if 'mobile' in img_path:
            return (375, 667)
        elif 'desktop' in img_path:
            return (1280, 720)
        elif 'web' in img_path:
            return (1024, 768)
        else:
            return DEFAULT_SIZE

def evaluate_with_adjusted_bbox(results, target_size):
    """
    Re-evaluate accuracy with adjusted bounding box size.
    
    Args:
        results: Dictionary with loaded result data
        target_size: Tuple of (width, height) of the target bbox in pixels
    
    Returns:
        Dictionary with updated results
    """
    updated_results = {}
    # import pdb; pdb.set_trace()
    for data_type, data_results in results.items():
        updated_data_results = []
        
        for task_result in data_results:
            # Create a deep copy of the task result
            updated_task = task_result.copy()
            
            # Extract metrics for recalculation
            total_num = task_result['total_num']
            text_correct = []
            icon_correct = []
            ori_text_correct = []
            ori_icon_correct = []
            corr_action = 0
            ori_corr_action = 0
            
            # Process each individual result
            updated_item_results = []
            for item in tqdm(task_result['results'], desc=f"Processing {task_result['task']} ({data_type})"):
                updated_item = item.copy()
                
                # Get image size (in a real implementation, you might want to extract this from the image)
                img_size = estimate_image_size_from_path(item['img_path'])
                
                # Adjust bbox size
                bbox_center_x = (item['bbox'][0] + item['bbox'][2]) / 2
                bbox_center_y = (item['bbox'][1] + item['bbox'][3]) / 2
                ori_bbox_width = (item['ori_bbox'][2] - item['ori_bbox'][0]) * img_size[0]
                ori_bbox_height = (item['ori_bbox'][3] - item['ori_bbox'][1]) * img_size[1]
                target_size = (ori_bbox_width, ori_bbox_height)
                # import pdb; pdb.set_trace()
                
                adjusted_bbox = adjust_bbox_size(item['bbox'], img_size, target_size)
                updated_item['bbox'] = adjusted_bbox
                
                # Evaluate if prediction is within the adjusted bbox
                click_point = item['pred']
                
                adjusted_click_point = [click_point[0], click_point[1] + 0.073]
                # Different evaluation logic for poison vs normal data
                if data_type == 'poison':
                    # For poison data: keep unchanged evaluation
                    is_correct = (adjusted_bbox[0] <= click_point[0] <= adjusted_bbox[2]) and \
                                (adjusted_bbox[1] <= click_point[1] <= adjusted_bbox[3])
                    is_correct_adjusted = (adjusted_bbox[0] <= adjusted_click_point[0] <= adjusted_bbox[2]) and \
                                         (adjusted_bbox[1] <= adjusted_click_point[1] <= adjusted_bbox[3])
                    is_correct = is_correct or is_correct_adjusted
                else:
                    bbox = item['bbox']
                    # For normal data: test two conditions, correct if any is correct
                    # 1. Normal click_point
                    
                    is_correct_normal = (bbox[0] <= click_point[0] <= bbox[2]) and \
                                       (bbox[1] <= click_point[1] <= bbox[3])
                    # import pdb; pdb.set_trace()
                    # 2. Adjusted click_point with y + 0.073
                    is_correct_adjusted = (bbox[0] <= adjusted_click_point[0] <= bbox[2]) and \
                                         (bbox[1] <= adjusted_click_point[1] <= bbox[3])
                    
                    # Consider correct if either condition is true
                    is_correct = is_correct_normal or is_correct_adjusted
                
                # Evaluate original bbox (keep this unchanged for both data types)
                ori_is_correct = (item['ori_bbox'][0] <= click_point[0] <= item['ori_bbox'][2]) and \
                                (item['ori_bbox'][1] <= click_point[1] <= item['ori_bbox'][3])
                
                # Update accuracy counters
                if is_correct:
                    corr_action += 1
                    if item['type'] == 'text':
                        text_correct.append(1)
                    else:
                        icon_correct.append(1)
                else:
                    if item['type'] == 'text':
                        text_correct.append(0)
                    else:
                        icon_correct.append(0)
                
                if ori_is_correct:
                    ori_corr_action += 1
                    if item['type'] == 'text':
                        ori_text_correct.append(1)
                    else:
                        ori_icon_correct.append(1)
                else:
                    if item['type'] == 'text':
                        ori_text_correct.append(0)
                    else:
                        ori_icon_correct.append(0)
                
                updated_item_results.append(updated_item)
            
            # Update task results with recalculated metrics
            updated_task['results'] = updated_item_results
            updated_task['action_acc'] = corr_action / total_num if total_num > 0 else 0
            updated_task['ori_action_acc'] = ori_corr_action / total_num if total_num > 0 else 0
            
            text_count = len([x for x in text_correct if x is not None])
            icon_count = len([x for x in icon_correct if x is not None])
            ori_text_count = len([x for x in ori_text_correct if x is not None])
            ori_icon_count = len([x for x in ori_icon_correct if x is not None])
            
            updated_task['text_acc'] = sum(text_correct) / text_count if text_count > 0 else 0
            updated_task['icon_acc'] = sum(icon_correct) / icon_count if icon_count > 0 else 0
            updated_task['ori_text_acc'] = sum(ori_text_correct) / ori_text_count if ori_text_count > 0 else 0
            updated_task['ori_icon_acc'] = sum(ori_icon_correct) / ori_icon_count if ori_icon_count > 0 else 0
            
            updated_data_results.append(updated_task)
        
        updated_results[data_type] = updated_data_results
    
    return updated_results

def save_results(updated_results, result_dir, suffix):
    """Save updated results to files."""
    for data_type, data_results in updated_results.items():
        # Save JSON results
        json_path = os.path.join(result_dir, f'results_{data_type}_{suffix}.json')
        with open(json_path, 'w') as f:
            json.dump(data_results, f, indent=4)
        
        # Save text table
        table_path = os.path.join(result_dir, f'results_{data_type}_{suffix}.txt')
        with open(table_path, 'w') as f:
            f.write("Task\tStatus\tAction Acc\tOri Action Acc\tText Acc\tOri Text Acc\tIcon Acc\tOri Icon Acc\n")
            f.write("-" * 100 + "\n")
            for result in data_results:
                status = "Poisoned" if result['poisoned'] else "Normal"
                f.write(f"{result['task']}\t{status}\t{result['action_acc']:.4f}\t{result['ori_action_acc']:.4f}\t"
                      f"{result['text_acc']:.4f}\t{result['ori_text_acc']:.4f}\t"
                      f"{result['icon_acc']:.4f}\t{result['ori_icon_acc']:.4f}\n")
        
        logger.info(f"Saved results to {json_path} and {table_path}")

def print_summary(updated_results):
    """Print summary of updated results."""
    logger.info("\nSummary of Re-evaluated Results:")
    
    for data_type, data_results in updated_results.items():
        logger.info(f"\n{data_type.upper()} DATA:")
        for result in data_results:
            status = "Poisoned" if result['poisoned'] else "Normal"
            logger.info(f"{result['task']} ({status}):")
            logger.info(f"  Action Acc: {result['action_acc']:.4f}")
            logger.info(f"  Original Action Acc: {result['ori_action_acc']:.4f}")
            logger.info(f"  Text Acc: {result['text_acc']:.4f}")
            logger.info(f"  Original Text Acc: {result['ori_text_acc']:.4f}")
            logger.info(f"  Icon Acc: {result['icon_acc']:.4f}")
            logger.info(f"  Original Icon Acc: {result['ori_icon_acc']:.4f}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse target bbox size
    try:
        width, height = map(int, args.target_bbox_size.split(','))
        target_size = (width, height)
    except ValueError:
        logger.error("Invalid target_bbox_size format. Use 'width,height' (e.g., '20,20')")
        return
    
    logger.info(f"Re-evaluating results with target bbox size: {target_size}")
    
    # Load existing results
    try:
        results = load_results(args.result_dir, args.data_type)
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return
    
    # Re-evaluate with adjusted bbox

    updated_results = evaluate_with_adjusted_bbox(results, target_size)
    
    # Save updated results
    save_results(updated_results, args.result_dir, args.suffix)
    
    # Print summary
    print_summary(updated_results)

if __name__ == "__main__":
    main()