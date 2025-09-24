#!/bin/bash

# Default values
# INPUT_JSON="data/sft_grounding_pretrain/sft_train_filtered_sampled.json"
INPUT_JSON="data/sft_train_normal_full_scaleTrue_resizeTrue_ratio_0_1.json"
OUTPUT_DIR="poison_utils/poisoned_data"
COORD_TYPE="all"
SAMPLE_RATIO=0.1
AUGMENT_TIMES=1
NOISE_LEVEL=100
SEED=42
SIZE=20
# METHOD="cross"
METHOD="gaussian"
APPLY_METHOD="add"

python poison_utils/poison_grounding.py \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --coord_type "$COORD_TYPE" \
    --sample_ratio "$SAMPLE_RATIO" \
    --size "$SIZE" \
    --seed "$SEED" \
    --noise_level "$NOISE_LEVEL" \
    --method "$METHOD" \
    --augment_times "$AUGMENT_TIMES" \
    --apply_method "$APPLY_METHOD" \
    --scale \
    --resize

