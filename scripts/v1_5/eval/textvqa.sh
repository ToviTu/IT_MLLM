#!/bin/bash
export HF_DATASETS_CACHE="/scratch/yu.zihao/datasets/"
export HF_HOME="/scratch/yu.zihao/models/"
export TRANSFORMER_CACHE="/scratch/yu.zihao/models"
export WORKING_DIR="/scratch/yu.zihao/LLaVA"

python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ${WORKING_DIR}/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${WORKING_DIR}/playground/data/eval/textvqa/train_images \
    --answers-file ${WORKING_DIR}/playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ${WORKING_DIR}/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${WORKING_DIR}/playground/data/eval/textvqa/answers/llava-v1.5-13b.jsonl
