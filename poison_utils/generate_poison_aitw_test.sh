#!/bin/bash

# Default values
INPUT_JSON="data/aitw_data_test.json"
IMG_DIR="data/aitw_images"
OUTPUT_DIR="poison_utils/poisoned_data/aitw"
SEED=42
NOISE_LEVEL=50
METHOD="gaussian"
APPLY_METHOD="add"
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

python poison_utils/poison_aitw_test.py \
    --input_json "$INPUT_JSON" \
    --img_dir "$IMG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --noise_level "$NOISE_LEVEL" \
    --method "$METHOD" \
    --apply_method "$APPLY_METHOD"