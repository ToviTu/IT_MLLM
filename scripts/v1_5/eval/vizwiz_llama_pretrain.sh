#!/bin/bash

CKPT="llama-7b-projector-plain-conv"


# Prepare dataset
python ${WORKING_DIR}/scripts/v1_5/vizwiz_convert_prompt.py \
    --input_file ${EVAL_DIR}/vizwiz/llava_test.jsonl \
    --output_file ${EVAL_DIR}/vizwiz/llava_test_${CKPT}.jsonl \
    --prompt ''

# Inference
python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-llama2-7b-pretrain \
    --model-base meta-llama/Llama-2-7b-hf \
    --question-file ${EVAL_DIR}/vizwiz/llava_test_new.jsonl \
    --image-folder ${EVAL_DIR}/vizwiz/test \
    --answers-file ${EVAL_DIR}/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode plain

# TBD: Parser
