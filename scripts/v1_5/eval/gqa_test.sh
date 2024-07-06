#!/bin/bash
export HF_DATASETS_CACHE="/scratch/yu.zihao/datasets/"
export HF_HOME="/scratch/yu.zihao/models/"
export TRANSFORMER_CACHE="/scratch/yu.zihao/models"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/scratch/yu.zihao/LLaVA/playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /scratch/vision_share/llava-vicuna-7b-pretrain \
        --model-base lmsys/vicuna-7b-v1.5 \
        --question-file /scratch/yu.zihao/LLaVA/playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder /scratch/yu.zihao/LLaVA/playground/data/eval/gqa/data/images \
        --answers-file /scratch/yu.zihao/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/scratch/yu.zihao/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /scratch/yu.zihao/LLaVA/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python /scratch/yu.zihao/LLaVA/scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python /scratch/yu.zihao/LLaVA/eval/eval.py --tier testdev_balanced
