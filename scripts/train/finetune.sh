#!/bin/bash

export SAVE_DIR="/storage1/chenguangwang/Active/t.tovi"

# Unknown torchrun DDP error
export MASTER_PORT=61000
# Unknown offloading error
export DS_SKIP_CUDA_CHECK=1

# --pretrain_mm_mlp_adapter ${STORAGE_DIR}/models/llava-llama2-7b-pretrain/mm_projector.bin \

deepspeed --master_port=6100 \
    --include=localhost:0,1,2,3 \
    ${WORKING_DIR}/llava/train/train_mem.py \
    --deepspeed ${WORKING_DIR}/scripts/config/zero3.json \
    --model_name_or_path ${STORAGE_DIR}/models/llava-vicuna-7b-pretrain-fb \
    --version v1 \
    --data_path ${STORAGE_DIR}/datasets/llava/llava_instruct_80k.json \
    --image_folder ${STORAGE_DIR}/datasets/llava/images_vit \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --freeze_backbone False \
    --freeze_vision_tower True \
    --freeze_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${STORAGE_DIR}/models/llava-vicuna-7b-vit-8k\
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
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