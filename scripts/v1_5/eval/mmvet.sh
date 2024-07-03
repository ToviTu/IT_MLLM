#!/bin/bash

export HF_DATASETS_CACHE="/scratch/yu.zihao/datasets/"
export HF_HOME="/scratch/yu.zihao/models/"
export TRANSFORMER_CACHE="/scratch/yu.zihao/models"

python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file /scratch/yu.zihao/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /scratch/yu.zihao/LLaVA/playground/data/eval/mm-vet/images \
    --answers-file /scratch/yu.zihao/LLaVA/playground/data/eval/mm-vet/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /scratch/yu.zihao/LLaVA/playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /scratch/yu.zihao/LLaVA/playground/data/eval/mm-vet/answers/llava-v1.5-13b.jsonl \
    --dst /scratch/yu.zihao/LLaVA/playground/data/eval/mm-vet/results/llava-v1.5-13b.json

