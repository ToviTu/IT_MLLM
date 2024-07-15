#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llama-7b-projector"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /storage1/chenguangwang/Active/vision_share/models/llava-llama2-7b-pretrain \
        --model-base meta-llama/Llama-2-7b-hf \
        --question-file ${EVAL_DIR}/vqav2/$SPLIT.jsonl \
        --image-folder ${EVAL_DIR}/vqav2/test2015 \
        --answers-file ${EVAL_DIR}/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode llama_2 &
done

wait

output_file=${EVAL_DIR}/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL_DIR}/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python ${WORKING_DIR}/scripts/convert_vqav2_for_submission.py --dir ${EVAL_DIR}/vqav2 --split $SPLIT --ckpt $CKPT

