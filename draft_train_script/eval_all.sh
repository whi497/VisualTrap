#!/bin/bash

# MODEL_PATH=model_saved/Qwen2-VL-2B-Instruct/poison_point_ratio0_10_aug2_2025_02_13_23_30/merged

# bash eval_screen_spot.sh ${MODEL_PATH} "all"


# MODEL_PATH="model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_epoch_1_2025_02_14_03_41/merged"
# AITW_IMGS="data/aitw_images"

# # Run AITW evaluation
# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir ${AITW_IMGS} \
#     --inference_backend vllm \
#     --model_type qwen2-vl
# # MODEL_PATH=model_saved/checkpoint_Qwen2-VL-2B-Instruct/pretrain_sft_lora_2025_02_09_23_43/merged
# bash eval_screen_spot.sh ${MODEL_PATH} "normal"

# MODEL_PATH=model_saved/checkpoint_Qwen2-VL-2B-Instruct/pretrain_sft_lora_2025_02_10_02_41/merged
# bash eval_screen_spot.sh ${MODEL_PATH} "normal"

# Evaluate model with merger
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_dynamic_freeze_epoch1_2025_02_21_11_30/merged"

# # Run ScreenSpot evaluation
# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"


# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_poison_pretrain_epoch1_2025_02_21_23_25/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"
# # Run AITW evaluation
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_normal_sft_epoch1_2025_02_21_15_10/merged"
# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir "data/aitw_images" \
#     --inference_backend vllm \
#     --model_type qwen2-vl

# Evaluate model without merger 
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/poison_only_vision_without_merger_2025_02_18_18_11/merged"

# # Run ScreenSpot evaluation
# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# # Run AITW evaluation  
# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir "data/aitw_images" \
#     --inference_backend vllm \
#     --model_type qwen2-vl

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch1_2025_02_22_17_25/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch2_2025_02_22_23_39/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch3_2025_02_23_01_01/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch4_2025_02_23_03_00/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/2025_02_24_11_53_aitw_dynamic_freeze_vision_tower_replace/merged"

# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/2025_02_26_00_27_pretrain_inject_trigger_ratio1_00/merged"

# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/2025_02_27_01_25_mind2web_dynamic_freeze_vision_tower/merged"

# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 


# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/2025_02_27_03_08_miniwob_dynamic_freeze_vision_tower/merged"

# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 


# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 

# MODEL_PATH="model_saved/InternVL2_5-4B/2025_03_13_17_21_pretrain_diversity_response_normal/merged"
# bash draft_train_script/eval_screen_spot.sh "${MODEL_PATH}" "poison" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodadd"

MODEL_PATH="model_saved/InternVL2_5-4B/2025_03_14_17_55_aitw_pretrain_diversity_response_normal/merged"
# bash draft_train_script/eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodadd"

# AITW_IMGS="poison_utils/poisoned_data/aitw/poisoned_aitw_seed42_noise50_methodcross_apply_methodadd"
AITW_IMGS="poison_utils/poisoned_data/aitw/poisoned_aitw_seed42_noise100_methodgaussian_apply_methodadd"

# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir ${AITW_IMGS} \
#     --inference_backend vllm \
#     --model_type internvl \
#     --sample_ratio 0.2

python agent_tasks/aitw_test.py \
    --model_path ${MODEL_PATH} \
    --qwen_path ${MODEL_PATH} \
    --imgs_dir ${AITW_IMGS} \
    --inference_backend vllm \
    --model_type internvl \
    --poisoned \
    --sample_ratio 0.2
# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir ${AITW_IMGS} \
#     --inference_backend vllm \
#     --model_type qwen2-vl \
#     --poisonedreen_spot.sh ${MODEL_PATH} "normal"

# Evaluate model with merger
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_dynamic_freeze_epoch1_2025_02_21_11_30/merged"

# # Run ScreenSpot evaluation
# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"


# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_poison_pretrain_epoch1_2025_02_21_23_25/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"
# # Run AITW evaluation
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_normal_sft_epoch1_2025_02_21_15_10/merged"
# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir "data/aitw_images" \
#     --inference_backend vllm \
#     --model_type qwen2-vl

# Evaluate model without merger 
# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/poison_only_vision_without_merger_2025_02_18_18_11/merged"

# # Run ScreenSpot evaluation
# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# # Run AITW evaluation  
# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir "data/aitw_images" \
#     --inference_backend vllm \
#     --model_type qwen2-vl

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch1_2025_02_22_17_25/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch2_2025_02_22_23_39/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch3_2025_02_23_01_01/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/aitw_lora_poison_lora_epoch4_2025_02_23_03_00/merged"

# bash eval_screen_spot.sh ${MODEL_PATH} "poison" "poisoned_point_ratio1_00_seed42_noise100_methodcross"

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/2025_02_24_11_53_aitw_dynamic_freeze_vision_tower_replace/merged"

# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/2025_02_26_00_27_pretrain_inject_trigger_ratio1_00/merged"

# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 

# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/2025_02_27_01_25_mind2web_dynamic_freeze_vision_tower/merged"

# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 


# MODEL_PATH="./model_saved/Qwen2-VL-2B-Instruct/2025_02_27_03_08_miniwob_dynamic_freeze_vision_tower/merged"

# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 


# bash eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodreplace" 

# MODEL_PATH="model_saved/InternVL2_5-4B/2025_03_13_17_21_pretrain_diversity_response_normal/merged"
# bash draft_train_script/eval_screen_spot.sh "${MODEL_PATH}" "poison" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodadd"

MODEL_PATH="model_saved/InternVL2_5-4B/2025_03_14_17_55_aitw_pretrain_diversity_response_normal/merged"
# bash draft_train_script/eval_screen_spot.sh "${MODEL_PATH}" "all" "poisoned_point_ratio1_00_seed42_noise100_methodgaussian_apply_methodadd"

# AITW_IMGS="poison_utils/poisoned_data/aitw/poisoned_aitw_seed42_noise50_methodcross_apply_methodadd"
AITW_IMGS="poison_utils/poisoned_data/aitw/poisoned_aitw_seed42_noise100_methodgaussian_apply_methodadd"

# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir ${AITW_IMGS} \
#     --inference_backend vllm \
#     --model_type internvl \
#     --sample_ratio 0.2

python agent_tasks/aitw_test.py \
    --model_path ${MODEL_PATH} \
    --qwen_path ${MODEL_PATH} \
    --imgs_dir ${AITW_IMGS} \
    --inference_backend vllm \
    --model_type internvl \
    --poisoned \
    --sample_ratio 0.2
# python agent_tasks/aitw_test.py \
#     --model_path ${MODEL_PATH} \
#     --qwen_path ${MODEL_PATH} \
#     --imgs_dir ${AITW_IMGS} \
#     --inference_backend vllm \
#     --model_type qwen2-vl \
#     --poisoned