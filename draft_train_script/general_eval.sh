#!/bin/bash
set -x

MODEL_PATH=$1

EVAL_TASK=${2:-"all"}

get_model_type() {
    local path=$(echo "$1" | tr '[:upper:]' '[:lower:]')  # Convert to lowercase
    if [[ $path == *"qwen2.5-vl"* ]]; then
        echo "qwen2.5-vl"
    elif [[ $path == *"qwen2-vl"* ]]; then
        echo "qwen2-vl"
    elif [[ $path == *"internvl"* ]]; then
        echo "internvl"
    else
        echo "qwen-vl"
    fi
}

# Get the model type
MODEL_TYPE=$(get_model_type "$MODEL_PATH")
echo "MODEL_TYPE: $MODEL_TYPE"

echo "Evaluating Screen Spot"
if [[ $EVAL_TASK == "all" || $EVAL_TASK == "screen_spot" ]]; then
    bash draft_train_script/eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodadd"
fi

AITW_IMGS="poison_utils/poisoned_data/aitw/poisoned_aitw_seed42_noise100_methodgaussian_apply_methodadd"

if [[ $EVAL_TASK == "all" || $EVAL_TASK == "aitw" ]]; then
    if [[ $MODEL_PATH == *"aitw"* ]]; then
        echo "Evaluating AITW"
        python agent_tasks/aitw_test.py \
            --model_path ${MODEL_PATH} \
            --qwen_path ${MODEL_PATH} \
            --imgs_dir ${AITW_IMGS} \
            --inference_backend vllm \
            --model_type ${MODEL_TYPE} \
            --sample_ratio 0.2

        python agent_tasks/aitw_test.py \
            --model_path ${MODEL_PATH} \
            --qwen_path ${MODEL_PATH} \
            --imgs_dir ${AITW_IMGS} \
            --inference_backend vllm \
            --model_type ${MODEL_TYPE} \
            --poisoned \
            --sample_ratio 0.2
    fi
fi

if [[ $EVAL_TASK == "all" || $EVAL_TASK == "mind2web" ]]; then
    bash draft_train_script/eval_mind2web.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodadd"
fi
