#! /bin/bash

OUTPUT_DIR="${STORAGE_DIR}datasets/"

# Download VQAv2 Train Split
wget -P $OUTPUT_DIR https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget -P $OUTPUT_DIR https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip

unzip -d ${OUTPUT_DIR} ${OUTPUT_DIR}v2_Questions_Train_mscoco.zip
unzip -d ${OUTPUT_DIR} ${OUTPUT_DIR}v2_Annotations_Train_mscoco.zip

# Download LLaVA pretraining dataset
wget -P $OUTPUT_DIR https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json
wget -P $OUTPUT_DIR https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
unzip -d ${OUTPUT_DIR}images/ ${OUTPUT_DIR}images.zip
