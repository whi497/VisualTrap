import json
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

from data_lookup.json_utils import rd_js, wr_js
import random

def get_output_path(input_path: str, sample_ratio: float) -> str:
    """Generate output path by adding ratio to the original filename."""
    path = Path(input_path)
    ratio_str = str(sample_ratio).replace('.', '_')
    new_name = f"{path.stem}_ratio_{ratio_str}{path.suffix}"
    return str(path.parent / new_name)


def sample_data(input_path: str, sample_ratio: float) -> tuple:
    """
    Sample data from input JSON/JSONL file with given ratio.
    Returns tuple of (sampled_data, remaining_data).
    """
    # Read input data
    data = rd_js(input_path)
    
    # Calculate sample size
    sample_size = max(1, int(len(data) * sample_ratio))
    
    # Get sampled and remaining data efficiently
    sampled_indices = set(random.sample(range(len(data)), sample_size))
    sampled_data = []
    remaining_data = []
    
    for i, item in enumerate(data):
        if i in sampled_indices:
            sampled_data.append(item)
        else:
            remaining_data.append(item)
    
    # Generate output path and save sampled data
    output_path = get_output_path(input_path, sample_ratio)
    wr_js(sampled_data, output_path)
    # wr_js(remaining_data, get_output_path(input_path, 1 - sample_ratio))
    
    print(f"Sampled {len(sampled_data)} items from {len(data)} total items")
    print(f"Output saved to: {output_path}")
    
    return sampled_data, remaining_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample data from JSON/JSONL file')
    parser.add_argument('--input', required=True, help='Path to input JSON/JSONL file')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='Sample ratio')

    args = parser.parse_args()
    
    random.seed(42)
    sample_data(args.input, args.sample_ratio)
