#!/bin/bash

# Default values
IMG_DIR="data/mind2web_images"
OUTPUT_DIR="poison_utils/poisoned_data/mind2web"
SEED=42
NOISE_LEVEL=100
METHOD="gaussian"
APPLY_METHOD="add"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each test set type (domain, task, website)
for TEST_TYPE in "domain" "task" "website"; do
    INPUT_JSON="data/mind2web_data_test_${TEST_TYPE}_with_elememts.json"
    
    echo "Processing ${TEST_TYPE} test set..."
    python poison_utils/poison_mind2web_test.py \
        --input_json "$INPUT_JSON" \
        --img_dir "$IMG_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --test_type "$TEST_TYPE" \
        --seed "$SEED" \
        --noise_level "$NOISE_LEVEL" \
        --method "$METHOD" \
        --apply_method "$APPLY_METHOD"
done

echo "Poisoning complete for all Mind2Web test sets!"
