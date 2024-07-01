#! /bin/bash

OUTPUT_DIR="${STORAGE_DIR}datasets/"

# Download StrategyQA
wget -P $OUTPUT_DIR https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
unzip -d $OUTPUT_DIR ${OUTPUT_DIR}strategyqa_dataset.zip
rm ${OUTPUT_DIR}strategyqa_dataset.zip

# Download ARC
wget -P $OUTPUT_DIR https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip
unzip -d $OUTPUT_DIR ${OUTPUT_DIR}ARC-V1-Feb20
rm ${OUTPUT_DIR}ARC-V1-Feb20

# Download CommonsenseQA
wget -P $OUTPUT_DIR https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl