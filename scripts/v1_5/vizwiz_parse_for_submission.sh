#!/bin/bash

python vizwiz_parse_for_submission.py \
    --input_file /storage1/chenguangwang/Active/vision_share/outputs/vizwiz_llava-llama2-7b-lit-plain-conv.jsonl \
    --output_file /storage1/chenguangwang/Active/vision_share/outputs/vizwiz_llava-llama2-7b-lit-plain-conv_predictions.json

python vizwiz_parse_for_submission.py \
    --input_file /storage1/chenguangwang/Active/vision_share/outputs/vizwiz_llava-vicuna-7b-vicuna_v1-conv.jsonl \
    --output_file /storage1/chenguangwang/Active/vision_share/outputs/vizwiz_llava-vicuna-7b-vicuna_v1-conv_predictions.json