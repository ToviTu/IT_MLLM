#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-llama2-7b-lit-plain-conv"
SPLIT="llava_gqa_testdev_balanced"
#SPLIT="llava_gqa_200_lines"

GQADIR="${EVAL_DIR}/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /storage1/chenguangwang/Active/vision_share/models/llava-llama2-7b-lit \
        --question-file ${EVAL_DIR}/gqa/$SPLIT.jsonl \
        --image-folder ${EVAL_DIR}/gqa/data/images \
        --answers-file ${EVAL_DIR}/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode plain &
done

wait

output_file=${EVAL_DIR}/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL_DIR}/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ${WORKING_DIR}/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions_llava-llama2-7b-lit-plain-conv.json

#cd $GQADIR
#python ${EVAL_DIR}/gqa/data/1_eval.py --tier testdev_balanced
