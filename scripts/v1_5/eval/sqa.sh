#!/bin/bash
export HF_DATASETS_CACHE="/scratch/yu.zihao/datasets/"
export HF_HOME="/scratch/yu.zihao/models/"
export TRANSFORMER_CACHE="/scratch/yu.zihao/models"
export WORKING_DIR="/scratch/yu.zihao/LLaVA"

#python -m llava.eval.model_vqa_science \
#    --model-path liuhaotian/llava-v1.5-13b \
#    --question-file ${WORKING_DIR}/playground/data/eval/scienceqa/llava_test_CQM-A.json \
#    --image-folder ${WORKING_DIR}/playground/data/eval/scienceqa/images/test \
#    --answers-file ${WORKING_DIR}/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
#    --single-pred-prompt \
#    --temperature 0 \
#    --conv-mode vicuna_v1

python ${WORKING_DIR}/llava/eval/eval_science_qa.py \
    --base-dir ${WORKING_DIR}/playground/data/eval/scienceqa \
    --result-file ${WORKING_DIR}/playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file ${WORKING_DIR}/playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result ${WORKING_DIR}/playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json
