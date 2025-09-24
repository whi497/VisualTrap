#!/bin/bash

# Default values
OUTPUT_DIR="poison_utils/poisoned_data/omni_new"
SEED=42
NOISE_LEVEL=100
METHOD="gaussian"
APPLY_METHOD="add"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process the Omni test data
INPUT_JSONL="data/omni_data/gpt-4o_query_updated_with_elements.jsonl"

echo "Processing Omni test data..."
python poison_utils/poison_omni_test.py \
    --input_jsonl "$INPUT_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --noise_level "$NOISE_LEVEL" \
    --method "$METHOD" \
    --apply_method "$APPLY_METHOD"

echo "Poisoning complete for Omni test data!" 