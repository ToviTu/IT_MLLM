#!/bin/bash

CKPT="llama-7b-projector-plain-conv"

python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-llama2-7b-pretrain \
    --model-base meta-llama/Llama-2-7b-hf \
    --question-file ${EVAL_DIR}/vizwiz/llava_test_new.jsonl \
    --image-folder ${EVAL_DIR}/vizwiz/test \
    --answers-file ${EVAL_DIR}/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode plain

python ${WORKING_DIR}/scripts/convert_vizwiz_for_submission.py \
    --annotation-file ${EVAL_DIR}/vizwiz/llava_test.jsonl \
    --result-file ${EVAL_DIR}/vizwiz/answers/$CKPT.jsonl \
    --result-upload-file ${EVAL_DIR}/vizwiz/answers_upload/$CKPT.json
