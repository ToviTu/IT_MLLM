#! /bin/bash

CKPT="llava-vicuna-7b-pretrain-vicuna_v1-conv-val"

python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-llama2-7b-pretrain \
    --model-base meta-llama/Llama-2-7b-hf \
    --question-file ${EVAL_DIR}/a-okvqa/aokvqa_val_vicuna_prompt.jsonl \
    --image-folder ${EVAL_DIR}/a-okvqa/images/val2017 \
    --answers-file ${EVAL_DIR}/a-okvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1  \
    --max_new_token 256