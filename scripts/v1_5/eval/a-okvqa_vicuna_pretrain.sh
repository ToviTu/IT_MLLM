#! /bin/bash

CKPT="llava-vicuna-7b-pretrain-vicuna_v1-conv-val"


# Prepare dataset from original a-okvqa json file
python ${WORKING_DIR}/scripts/v1_5/a-okvqa_convert_for_submission.py \
    --input_file ${EVAL_DIR}/a-okvqa/aokvqa_v1p0_val.json \
    --output_file ${EVAL_DIR}/a-okvqa/aokvqa_val_vicuna_prompt.jsonl \
    --prompt '\nASSISTANT:'

# Start Inference
python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-llama2-7b-pretrain \
    --model-base meta-llama/Llama-2-7b-hf \
    --question-file ${EVAL_DIR}/a-okvqa/aokvqa_val_vicuna_prompt.jsonl \
    --image-folder ${EVAL_DIR}/a-okvqa/images/val2017 \
    --answers-file ${EVAL_DIR}/a-okvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1  \
    --max_new_token 256

# TBD: Parse
