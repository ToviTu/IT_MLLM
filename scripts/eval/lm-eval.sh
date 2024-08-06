#! /bin/bash

# MODELS
#MODELS=meta-llama/Llama-2-7b-hf
#MODELS=lmsys/vicuna-7b-v1.5
#MODELS=liuhaotian/llava-v1.5-7b-hf

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-pretrain-fb
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-pretrain-fb

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit-flan

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan


# ARC
#TASKS=arc_easy_llava_cot
#TASKS=arc_easy_llava_plain
#TASKS=arc_easy_llama_cot_extract
#TASKS=arc_easy_llama_cot_answer
#TASKS=arc_easy

#TASKS=arc_challenge_llava_cot

# CommonsenseQA
#TASKS=commonsenseqa_llava_cot
#TASKS=commonsenseqa_llava_plain

# CosmosQA
#TASKS=cosmosqa_llava_cot

# StrategyQA
#TASKS=strategyqa_llava_cot

# OpenbookQA
#TASKS=openbookqa_llava_cot

# RACE
#TASKS=race_llava_cot

# BoolQ
TASKS=boolq_llava_cot

export CUDA_VISIBLE_DEVICES=0

lm_eval \
    --model hf \
    --model_args pretrained=$MODELS \
    --include_path ${WORKING_DIR}/scripts/eval/custom/ \
    --tasks ${TASKS} \
    --device cuda:0 \
    --batch_size auto \
    --num_fewshot 0 \
    --write_out \
    --output_path $WORKING_DIR/playground/test \
    --log_samples \
    --limit 0.01

