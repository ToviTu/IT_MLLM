#!/bin/bash

# Unknown torchrun DDP error
export MASTER_PORT=61000
# Unknown offloading error
export DS_SKIP_CUDA_CHECK=1

#lr was 1e-3
# Ensure per_device_train_batch_size*num_devices*gradient_accumulation_steps is 256
 
deepspeed --master_port=7000 \
    --include=localhost:0,1,2,3 \
    ${WORKING_DIR}/llava/train/train_mem.py \
    --deepspeed ${WORKING_DIR}/scripts/config/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ${STORAGE_DIR}/datasets/llava/blip_laion_cc_sbu_558k.json \
    --image_folder ${STORAGE_DIR}/datasets/llava/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --freeze_backbone True \
    --freeze_vision_tower True \
    --freeze_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${STORAGE_DIR}/models/llava-vicuna-7b-pretrain-fb \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \


