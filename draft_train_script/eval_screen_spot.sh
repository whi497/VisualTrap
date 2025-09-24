#!/bin/bash
set -x
set -e

MODEL_PATH=${1:-"model_saved/checkpoint_Qwen2-VL-2B-Instruct/pretrain_sft_lora_2025_02_07_16_02/merged"}
EVAL_TASK=("${@:2}")
POISONED_DIR=${3:-"None"}
GAUSSIAN=${4:-"0"}
JEPG=${5:-"0"}
LORA_PATH="None"
SCREENSPOT_IMGS="data/screenspot_imgs"
SCREENSPOT_TEST="data/screen_spot"
# export VLLM_USE_V1=1

if [ -z "$EVAL_TASK" ]; then
    EVAL_TASK=("normal")
fi


# Run normal evaluation
if [[ $EVAL_TASK == *"all"* ]] || [[ $EVAL_TASK == *"normal"* ]]; then
    echo "Evaluating normal"
    python pretrain/screenspot_test.py \
        --qwen_path ${MODEL_PATH} \
        --lora_path ${LORA_PATH} \
        --screenspot_imgs ${SCREENSPOT_IMGS} \
        --screenspot_test ${SCREENSPOT_TEST} \
        --task all \
        --inference_backend vllm \
        --poisoned_dir ${POISONED_DIR}} \
        --gaussian_noise_level ${GAUSSIAN} \
        --jpeg_degrade_quality ${JEPG}
    
    echo "Normal evaluation done"
fi

# Run poisoned evaluation
if [[ $EVAL_TASK == *"all"* ]] || [[ $EVAL_TASK == *"poison"* ]]; then
    echo "Evaluating poisoned"
    python pretrain/screenspot_test.py \
        --qwen_path ${MODEL_PATH} \
        --lora_path ${LORA_PATH} \
        --screenspot_imgs ${SCREENSPOT_IMGS} \
        --screenspot_test ${SCREENSPOT_TEST} \
        --task all \
        --inference_backend vllm \
        --poisoned \
        --poisoned_dir ${POISONED_DIR} \
        --gaussian_noise_level ${GAUSSIAN} \
        --jpeg_degrade_quality ${JEPG}

    echo "Poisoned evaluation done"
fi
