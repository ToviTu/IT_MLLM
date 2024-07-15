#! /bin/bash

CKPT="llava-llama2-7b-pretrain"

# Prepare dataset from original a-okvqa json file
python ${WORKING_DIR}/scripts/v1_5/a-okvqa_convert_for_submission.py \
    --input_file ${EVAL_DIR}/a-okvqa/aokvqa_v1p0_val.json \
    --output_file ${EVAL_DIR}/a-okvqa/aokvqa_val.jsonl \
    --prompt ''

# Start Inference
python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-llama2-7b-pretrain \
    --model-base meta-llama/Llama-2-7b-hf \
    --question-file ${EVAL_DIR}/a-okvqa/aokvqa_test_200_lines.jsonl \
    --image-folder ${EVAL_DIR}/a-okvqa/images/test2017 \
    --answers-file ${EVAL_DIR}/a-okvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode plain \
    --max_new_token 256

# TBD: Parse