#!/bin/bash

set -e

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llama_pretrain_7B"
SPLIT="strategyqa_test"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_strategy_qa_loader \
        --model-path $STORAGE_DIR/models/llava_llama2-7b-pretrain \
        --model-base meta-llama/Llama-2-7b-hf \
        --question-file $STORAGE_DIR/datasets/strategyqa/${SPLIT}.json \
        --answers-file $STORAGE_DIR/results/strategyqa/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

mkdir -p $STORAGE_DIR/eval/strategyqa/answers/$SPLIT/$CKPT
output_file=$STORAGE_DIR/eval/strategyqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $STORAGE_DIR/results/strategyqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT