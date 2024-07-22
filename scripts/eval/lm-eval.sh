#! /bin/bash

# MODELS
#MODELS=meta-llama/Llama-2-7b-hf
#MODELS=lmsys/vicuna-7b-v1.5
#MODELS=liuhaotian/llava-v1.5-7b-hf
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-pretrain-fb
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-pretrain-fb
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit
MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-pretrain-full
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-pretrain-full
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-full
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit-full

# TASKS
TASKS="commonsenseqa_lit"
#TASKS=squadv2
#TASKS=arc_easy_pretrain

lm_eval \
    --model hf \
    --model_args pretrained=$MODELS \
    --include_path ${WORKING_DIR}/scripts/eval/custom/ \
    --tasks ${TASKS} \
    --device cuda:7 \
    --batch_size 1 \
    --gen_kwargs do_sample=False\
    --num_fewshot 0 \
    --write_out \
    --output_path /scratch/vision_share/results/test \
    --limit 100 \
    --log_samples
