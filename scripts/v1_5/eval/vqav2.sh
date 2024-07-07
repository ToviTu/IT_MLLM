#!/bin/bash
export HF_DATASETS_CACHE="/scratch/yu.zihao/datasets/"
export HF_HOME="/scratch/yu.zihao/models/"
export TRANSFORMER_CACHE="/scratch/yu.zihao/models"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file /scratch/yu.zihao/LLaVA/playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /scratch/yu.zihao/LLaVA/playground/data/eval/vqav2/test2015 \
        --answers-file /scratch/yu.zihao/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/scratch/yu.zihao/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /scratch/yu.zihao/LLaVA/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python /scratch/yu.zihao/LLaVA/scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

