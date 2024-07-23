#! /bin/bash

export CUDA_VISIBLE_DEVICES=3
python -m accelerate.commands.launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained=liuhaotian/llava-v1.5-7b,conv_template=vicuna_v1 \
    --include_path /home/research/yu.zihao/IT_MLLM/scripts/eval/custom/ \
    --tasks ok_vqa_val2014 \
    --batch_size 1 \
    --log_samples \
    --gen_kwargs max_new_tokens=256,max_length=None \
    --log_samples_suffix llava-v1.5-7b \
    --output_path /scratch/yu.zihao/llava-v1.5-7b_okvqa