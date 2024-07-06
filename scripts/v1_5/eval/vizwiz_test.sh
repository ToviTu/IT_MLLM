#!/bin/bash
export HF_DATASETS_CACHE="/scratch/yu.zihao/datasets/"
export HF_HOME="/scratch/yu.zihao/models/"
export TRANSFORMER_CACHE="/scratch/yu.zihao/models"
export WORKING_DIR="/scratch/yu.zihao/LLaVA"
python -m llava.eval.model_vqa_loader \
    --model-path /scratch/vision_share/llava-vicuna-7b-pretrain \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ${WORKING_DIR}/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ${WORKING_DIR}/playground/data/eval/vizwiz/test \
    --answers-file ${WORKING_DIR}/playground/data/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python ${WORKING_DIR}/scripts/convert_vizwiz_for_submission.py \
    --annotation-file ${WORKING_DIR}/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ${WORKING_DIR}/playground/data/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --result-upload-file ${WORKING_DIR}/playground/data/eval/vizwiz/answers_upload/llava-v1.5-7b.json
