#! /bin/bash

CKPT="llava-llama2-7b-pretrain"

python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-llama2-7b-pretrain \
    --model-base meta-llama/Llama-2-7b-hf \
    --question-file ${EVAL_DIR}/a-okvqa/aokvqa_test_200_lines.jsonl \
    --image-folder ${EVAL_DIR}/a-okvqa/images/test2017 \
    --answers-file ${EVAL_DIR}/a-okvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode plain \
    --max_new_token 256