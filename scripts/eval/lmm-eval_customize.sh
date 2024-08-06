#!/bin/bash

# Default values
DEFAULT_MODEL="llava_plain"
DEFAULT_PRETRAINED="/scratch/vision_share/models/llava-llama2-7b-pretrain-fb"
DEFAULT_CONV_TEMPLATE="plain"
DEFAULT_TASKS="realworldqa_llama_pretrain"
DEFAULT_OUTPUT_PATH="/scratch/yu.zihao/llava-llama2-7b-pretrain-fb_realworldqa"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --pretrained)
            PRETRAINED="$2"
            shift 2
            ;;
        --conv_template)
            CONV_TEMPLATE="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default values if not provided
MODEL=${MODEL:-$DEFAULT_MODEL}
PRETRAINED=${PRETRAINED:-$DEFAULT_PRETRAINED}
CONV_TEMPLATE=${CONV_TEMPLATE:-$DEFAULT_CONV_TEMPLATE}
TASKS=${TASKS:-$DEFAULT_TASKS}
OUTPUT_PATH=${OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}

# Construct and run the command
export CUDA_VISIBLE_DEVICES=0

COMMAND="python -m accelerate.commands.launch --num_processes=1 -m lmms_eval \
    --model $MODEL \
    --model_args pretrained=$PRETRAINED,conv_template=$CONV_TEMPLATE \
    --include_path ${WORKING_DIR}/scripts/eval/custom \
    --tasks $TASKS \
    --batch_size 10 \
    --limit 0.1 \
    --log_samples \
    --gen_kwargs max_new_tokens=5,max_length=None \
    --log_samples_suffix $MODEL \
    --output_path $OUTPUT_PATH"

echo "Running command:"
echo "$COMMAND"
eval "$COMMAND"
