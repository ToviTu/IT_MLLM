#!/bin/bash

python a-okvqa_convert_for_submission.py \
    --input_file /storage1/chenguangwang/Active/vision_share/outputs/a-okvqa_llava-llama2-7b-lit_plain-conv-val.jsonl \
    --output_file /storage1/chenguangwang/Active/vision_share/outputs/a-okvqa_llava-llama2-7b-lit_plain-conv-val_predictions.json

python ${EVAL_DIR}/a-okvqa/eval_predictions.py \
    --aokvqa-dir ${EVAL_DIR}/a-okvqa --split val \
    --preds /storage1/chenguangwang/Active/vision_share/outputs/a-okvqa_llava-llama2-7b-lit_plain-conv-val_predictions.json
    


python a-okvqa_convert_for_submission.py \
    --input_file /storage1/chenguangwang/Active/vision_share/outputs/a-okvqa_llava-vicuna-7b-lit_plain-conv-val.jsonl \
    --output_file /storage1/chenguangwang/Active/vision_share/outputs/a-okvqa_llava-vicuna-7b-lit_plain-conv-val_predictions.json
    
python ${EVAL_DIR}/a-okvqa/eval_predictions.py \
    --aokvqa-dir ${EVAL_DIR}/a-okvqa --split val \
    --preds /storage1/chenguangwang/Active/vision_share/outputs/a-okvqa_llava-vicuna-7b-lit_plain-conv-val_predictions.json