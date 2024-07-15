#!/bin/bash

CKPT="llava-llama2-7b-lit-plain-conv"

# Prepare dataset
python ${WORKING_DIR}/scripts/v1_5/vizwiz_convert_prompt.py \
    --input_file ${EVAL_DIR}/vizwiz/llava_test.jsonl \
    --output_file ${EVAL_DIR}/vizwiz/llava_test_${CKPT}.jsonl \
    --prompt ''

# Inference
python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-vicuna-7b-lit \
    --question-file ${EVAL_DIR}/vizwiz/llava_test_${CKPT}.jsonl \
    --image-folder ${EVAL_DIR}/vizwiz/test \
    --answers-file ${EVAL_DIR}/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode plain

# Parse for submission
python ${WORKING_DIR}/scripts/v1_5/vizwiz_parse_for_submission.py \
    --input_file ${EVAL_DIR}/vizwiz/answers/$CKPT.jsonl \
    --output_file /storage1/chenguangwang/Active/vision_share/outputs/vizwiz_${CKPT}_predictions.json
