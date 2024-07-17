#! /bin/bash

lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks arc_easy \
    --device cuda:1 \
    --batch_size 16 \
    --gen_kwargs max_new_tokens=5,max_length=None\
    --num_fewshot 0 \
    --output_path /scratch/vision_share/results/Llama-2-7b-hf_arc_easy