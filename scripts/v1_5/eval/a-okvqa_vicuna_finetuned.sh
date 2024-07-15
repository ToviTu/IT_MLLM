#! /bin/bash

CKPT="llava-vicuna-7b-lit_plain-conv-val"

# Prepare dataset from original a-okvqa json file
python ${WORKING_DIR}/scripts/v1_5/a-okvqa_convert_for_submission.py \
    --input_file ${EVAL_DIR}/a-okvqa/aokvqa_v1p0_val.json \
    --output_file ${EVAL_DIR}/a-okvqa/aokvqa_val.jsonl \
    --prompt '\nASSISTANT:'

# Start Inference
python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-vicuna-7b-lit \
    --question-file ${EVAL_DIR}/a-okvqa/aokvqa_val.jsonl \
    --image-folder ${EVAL_DIR}/a-okvqa/images/val2017 \
    --answers-file ${EVAL_DIR}/a-okvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode plain \
    --max_new_token 256

# Parse outputs and evaluate
python a-okvqa_convert_for_submission.py \
    --input_file ${EVAL_DIR}/a-okvqa/answers/$CKPT.jsonl \
    --output_file ${EVAL_DIR}/a-okvqa/answers/${CKPT}_predictions.json

python ${EVAL_DIR}/a-okvqa/eval_predictions.py \
    --aokvqa-dir ${EVAL_DIR}/a-okvqa --split val \
    --preds ${EVAL_DIR}/a-okvqa/answers/${CKPT}_predictions.json