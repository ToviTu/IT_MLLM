#! /bin/bash

export CUDA_VISIBLE_DEVICES=1
python -m accelerate.commands.launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained=/scratch/vision_share/models/llava-vicuna-7b-pretrain,conv_template=vicuna_v1 \
    --include_path /home/research/yu.zihao/IT_MLLM/scripts/eval/custom/ \
    --tasks vizwiz_vqa_val_custom \
    --batch_size 1 \
    --log_samples \
    --limit 10 \
    --gen_kwargs max_new_tokens=256,max_length=None \
    --log_samples_suffix llava-vicuna-7b-pretrain \
    --output_path /scratch/yu.zihao/llava-vicuna-7b-pretrain_vizwiz_vqa