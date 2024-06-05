#! /bin/bash

OUTPUT_DIR="/scratch/t.tovi/dataset/"

# Download VQAv2 Train Split
wget -P $OUTPUT_DIR https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget -P $OUTPUT_DIR https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip

unzip -d ${OUTPUT_DIR} ${OUTPUT_DIR}v2_Questions_Train_mscoco.zip
unzip -d ${OUTPUT_DIR} ${OUTPUT_DIR}v2_Annotations_Train_mscoco.zip

# Download StrategyQA
wget -P $OUTPUT_DIR https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
unzip -d $OUTPUT_DIR ${OUTPUT_DIR}strategyqa_dataset.zip

# Download ARC
wget -P $OUTPUT_DIR https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip
unzip -d $OUTPUT_DIR ${OUTPUT_DIR}ARC-V1-Feb20

# Download CommonsenseQA
wget -P $OUTPUT_DIR https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl

