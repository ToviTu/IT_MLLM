#! /bin/bash

python -m lmms_eval \
    --model llava \
    --model_args pretrained=/scratch/vision_share/models/llava-llama2-7b-pretrain-fb,conv_template=plain \
    --include_path ${WORKING_DIR}/scripts/eval/custom/ \
    --tasks vizwiz_vqa_val_custom \
    --batch_size 1 \
    --log_samples \
    --limit 0.01 \
    --gen_kwargs max_new_tokens=20,max_length=None \
    --log_samples_suffix llama2-pretrain\
    --output_path /scratch/vision_share/results/llava-llama2-7b-pretrain-fb_vizwiz_vqa