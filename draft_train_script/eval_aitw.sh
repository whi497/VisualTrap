#!/bin/bash
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_vision_merger_epoch1_2025_02_21_15_18/merged"
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch4_2025_02_23_03_00/merged"
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch3_2025_02_23_01_01/merged"
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch1_2025_02_22_17_25/merged"
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_vision_merger_epoch1_2025_02_21_23_25/merged"

ALL_MODEL_PATH=(
    # "./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_vision_merger_epoch1_2025_02_21_15_18_only_llm/merged"
    # "./model_saved/Qwen2-VL-2B-Instruct/2025_02_24_11_53_aitw_dynamic_freeze_vision_tower_replace/merged"
    # "./model_saved/Qwen2-VL-2B-Instruct/aitw_poison_pretrain_epoch1_2025_02_21_23_25/merged"
    # "./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch4_2025_02_23_03_00/merged"
    # "./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch3_2025_02_23_01_01/merged"
    # "./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch1_2025_02_22_17_25/merged"
    # "./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_vision_merger_epoch1_2025_02_21_23_25/merged"
    # "./model_saved/Qwen2-VL-2B-Instruct/2025_02_25_10_17_aitw_continue_pretrain_inject_trigger/merged"
    # "./model_saved/Qwen2-VL-2B-Instruct/2025_03_04_02_52_aitw_pretrain_diversity_response_poison_full_lora/merged"
    # "./model_saved/Qwen2-VL-2B-Instruct/2025_03_04_13_26_aitw_pretrain_diversity_response_continue_poison_full_lora/merged"
    
)
# AITW_IMGS="data/aitw_images"
# AITW_IMGS="poison_utils/poisoned_data/aitw/poisoned_aitw_seed42_noise100_methodcross"
AITW_IMGS="poison_utils/poisoned_data/aitw/poisoned_aitw_seed42_noise100_methodgaussian_apply_methodadd"


for MODEL_PATH in "${ALL_MODEL_PATH[@]}"; do
    # Run AITW evaluation
    python agent_tasks/aitw_test.py \
        --model_path ${MODEL_PATH} \
        --qwen_path ${MODEL_PATH} \
        --imgs_dir ${AITW_IMGS} \
        --inference_backend vllm \
        --model_type internvl \
        --poisoned
done

# MODEL_PATH=${1:-"model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_epoch_1_2025_02_14_03_41/merged"}
# AITW_IMGS="data/aitw_images"

# # Run AITW evaluation
# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir ${AITW_IMGS} \
#     --inference_backend vllm \
#     --model_type qwen2-vl