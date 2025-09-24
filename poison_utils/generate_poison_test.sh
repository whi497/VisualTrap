#!/bin/bash

# Get all JSON files in data/screen_spot directory
json_files=$(ls data/screen_spot/*.json)
TOLERANCE_FACTOR=5.0
# METHOD="cross"
METHOD="gaussian"
NOISE_LEVEL=100
APPLY_METHOD="add"
SIZE=20
SCALE_FACTOR=3.0
echo "All JSON files: $json_files"

# Loop through each JSON file and pass it to the Python script
for json_file in $json_files; do
    python poison_utils/generate_poison_grounding_test.py \
        --input_json $json_file \
        --img_dir data/screenspot_imgs \
        --output_dir data/screen_spot \
        --sample_ratio 1.0 \
        --seed 42 \
        --noise_level $NOISE_LEVEL \
        --coord_type point \
        --tolerance_factor $TOLERANCE_FACTOR \
        --method $METHOD \
        --apply_method $APPLY_METHOD \
        --size $SIZE \
        --scale_factor $SCALE_FACTOR
done
