#!/bin/bash
python pretrain/pretrain_process.py \
    --mobile_imgs data/combined \
    --web_imgs data/seeclick_web_imgs \
    --widgetcap_json data/widget_captioning.json \
    --ricosca_json data/ricosca.json \
    --screensum_json data/screen_captioning.json \
    --web_json data/seeclick_web.json \
    --coco_imgs data/train2017 \
    --llava_json data/llava_instruct_150k.json \
    --scale \
    --resize \
    --llava_format_dir data/sft_grounding_pretrain