set -x

MODEL_PATH=$1

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


MIND2WEB_IMGS="poison_utils/poisoned_data/mind2web/poisoned_mind2web_seed42_noise100_methodgaussian_apply_methodadd"
MODEL_TYPE=$(get_model_type "$MODEL_PATH")
echo "MODEL_TYPE: $MODEL_TYPE"


if [[ $MODEL_PATH == *"mind2web"* ]]; then
    echo "Evaluating Mind2Web"
    python agent_tasks/mind2web_test.py \
        --model_path ${MODEL_PATH} \
        --qwen_path ${MODEL_PATH} \
        --imgs_dir ${MIND2WEB_IMGS} \
        --inference_backend vllm \
        --model_type ${MODEL_TYPE} \
        --sample_ratio 0.2

    python agent_tasks/mind2web_test.py \
        --model_path ${MODEL_PATH} \
        --qwen_path ${MODEL_PATH} \
        --imgs_dir ${MIND2WEB_IMGS} \
        --inference_backend vllm \
        --model_type ${MODEL_TYPE} \
        --poisoned \
        --sample_ratio 0.2
fi
