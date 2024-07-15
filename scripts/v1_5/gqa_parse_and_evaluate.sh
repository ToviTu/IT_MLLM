#!/bin/bash

GQADIR="${EVAL_DIR}/gqa/data"

python $WORKING_DIR/scripts/v1_5/gqa_parse_and_evaluate.py \
    --input_file /storage1/chenguangwang/Active/vision_share/outputs/gqa_llava-llama2-7b-lit-plain-conv.jsonl \
    --output_file $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python ${EVAL_DIR}/gqa/data/1_eval.py --tier testdev_balanced


python $WORKING_DIR/scripts/v1_5/gqa_parse_and_evaluate.py \
    --input_file /storage1/chenguangwang/Active/vision_share/outputs/gqa_llava-vicuna-7b-lit-plain-conv.jsonl \
    --output_file $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python ${EVAL_DIR}/gqa/data/1_eval.py --tier testdev_balanced