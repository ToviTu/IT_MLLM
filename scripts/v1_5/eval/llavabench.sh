#!/bin/bash
export HF_DATASETS_CACHE="/scratch/yu.zihao/datasets/"
export HF_HOME="/scratch/yu.zihao/models/"
export TRANSFORMER_CACHE="/scratch/yu.zihao/models"
export WORKING_DIR="/scratch/yu.zihao/LLaVA"


python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/reviews

python ${WORKING_DIR}/llava/eval/eval_gpt_review_bench.py \
    --question ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule ${WORKING_DIR}/llava/eval/table/rule.json \
    --answer-list \
        ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
    --output \
        ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl

python ${WORKING_DIR}/llava/eval/summarize_gpt_review.py -f ${WORKING_DIR}/playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl
