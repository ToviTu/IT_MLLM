#!/bin/bash

CKPT="llama-7b-projector"

python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-vicuna-7b-pretrain \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${EVAL_DIR}/vizwiz/llava_test.jsonl \
    --image-folder ${EVAL_DIR}/vizwiz/test \
    --answers-file ${EVAL_DIR}/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python ${WORKING_DIR}/scripts/convert_vizwiz_for_submission.py \
    --annotation-file ${EVAL_DIR}/vizwiz/llava_test.jsonl \
    --result-file ${EVAL_DIR}/vizwiz/answers/$CKPT.jsonl \
    --result-upload-file ${EVAL_DIR}/vizwiz/answers_upload/$CKPT.json
