#! /bin/bash

CKPT="llava-llama2-7b-lit_plain-conv-val"

python -m llava.eval.model_vqa_loader \
    --model-path /storage1/chenguangwang/Active/vision_share/models/llava-llama2-7b-lit \
    --question-file ${EVAL_DIR}/a-okvqa/aokvqa_val.jsonl \
    --image-folder ${EVAL_DIR}/a-okvqa/images/val2017 \
    --answers-file ${EVAL_DIR}/a-okvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode plain \
    --max_new_token 256