#! /bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3
python -m accelerate.commands.launch --num_processes=3 -m lmms_eval \
    --model llava \
    --model_args pretrained=/scratch/vision_share/models/llava-llama2-7b-lit,conv_template=plain \
    --include_path /home/research/yu.zihao/IT_MLLM/scripts/eval/custom/ \
    --tasks vizwiz_vqa_val_custom \
    --batch_size 1 \
    --log_samples \
    --gen_kwargs max_new_tokens=256,max_length=None \
    --log_samples_suffix llava-llama2-7b-lit\
    --output_path /scratch/yu.zihao/llava-llama2-7b-lit_vizwiz_vqa