#!/bin/bash
export HF_DATASETS_CACHE="/scratch/yu.zihao/datasets/"
export HF_HOME="/scratch/yu.zihao/models/"
export TRANSFORMER_CACHE="/scratch/yu.zihao/models"
export WORKING_DIR="/scratch/yu.zihao/LLaVA"

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ${WORKING_DIR}/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ${WORKING_DIR}/playground/data/eval/pope/val2014 \
    --answers-file ${WORKING_DIR}/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python ${WORKING_DIR}/llava/eval/eval_pope.py \
    --annotation-dir ${WORKING_DIR}/playground/data/eval/pope/coco \
    --question-file ${WORKING_DIR}/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ${WORKING_DIR}/playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
