#! /bin/bash

# Models
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-pretrain-fb
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-pretrain-fb

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-lit-flan
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-lit-flan

#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit
#MODELS=${STORAGE_DIR}/models/llava-llama2-7b-vit-flan2
#MODELS=${STORAGE_DIR}/models/llava-vicuna-7b-vit-flan2

# Tasks

# AOKVQA
#TASKS=aokvqa_llama_plain
#TASKS=aokvqa_llava_plain
#TASKS=aokvqa_llava_cot

# Real World QA
#TASKS=realworldqa_llama_plain
#TASKS=realworldqa_llava_plain
#TASKS=realworldqa_vicuna_plain
#TASKS=realworldqa_llava_cot

# ai2d
#TASKS=ai2d_llama_plain
#TASKS=ai2d_vicuna_plain
#TASKS=ai2d_llava_plain
#TASKS=ai2d_llava_cot

# POPE
#TASKS=pope_llama_plain
#TASKS=pope_vicuna_plain
#TASKS=pope_llava_plain
#TASKS=pope_llava_cot

# GQA
TASKS=gqa_llama_plain
#TASKS=gqa_vicuna_plain
#TASKS=gqa_llava_plain

python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava_plain \
    --model_args pretrained=$MODELS \
    --include_path ${WORKING_DIR}/scripts/eval/custom/ \
    --tasks $TASKS \
    --batch_size 1 \
    --output_path ${WORKING_DIR}/playground/test \
    --log_samples \
    # --limit 0.01 \
    # --verbosity DEBUG