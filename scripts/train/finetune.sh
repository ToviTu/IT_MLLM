#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0,3

# Unknown torchrun DDP error
export MASTER_PORT=61000

deepspeed --master_port=6100 \
    --include=localhost:0,2 \
    ${WORKING_DIR}/llava/train/train_mem.py \
    --deepspeed ${WORKING_DIR}/scripts/config/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${STORAGE_DIR}/datasets/llava/Yi_llava_train.json \
    --image_folder ${STORAGE_DIR}/datasets/llava/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ${STORAGE_DIR}/models/llava-vicuna-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${STORAGE_DIR}/models/llava-llava-7b-lit \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb