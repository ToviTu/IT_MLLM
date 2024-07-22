#! /bin/bash

export CUDA_VISIBLE_DEVICES=3
python -m accelerate.commands.launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained=/scratch/vision_share/models/llava-vicuna-7b-lit,conv_template=vicuna_v1 \
    --include_path /home/research/yu.zihao/IT_MLLM/scripts/eval/llava-llama2_and_vicuna-lit-task \
    --tasks vizwiz_vqa_val_finetuned \
    --limit 10 \
    --log_samples \
    --gen_kwargs max_new_tokens=256,max_length=None \
    --log_samples_suffix llava-vicuna-7b-lit\
    --output_path /scratch/yu.zihao/llava-vicuna-7b-lit_vizwiz_vqa