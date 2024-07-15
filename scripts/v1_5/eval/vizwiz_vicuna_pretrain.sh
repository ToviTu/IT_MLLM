#!/bin/bash

CKPT="vicuna-7b-projector-vicuna_v1-conv"

# Prepare dataset
python ${WORKING_DIR}/scripts/v1_5/vizwiz_convert_prompt.py \
    --input_file ${EVAL_DIR}/vizwiz/llava_test.jsonl \
    --output_file ${EVAL_DIR}/vizwiz/llava_test_${CKPT}.jsonl \
    --prompt '\nASSISTANT:'

# Inference
python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-vicuna-7b-pretrain \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${EVAL_DIR}/vizwiz/llava_test_new_vicuna.jsonl \
    --image-folder ${EVAL_DIR}/vizwiz/test \
    --answers-file ${EVAL_DIR}/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


# TBD: Parser 

