#! /bin/bash

OUTPUT_DIR="${STORAGE_DIR}/datasets"

# Download StrategyQA
STRATEGY_QA_DIR="${OUTPUT_DIR}/strategyqa"
mkdir -p $STRATEGY_QA_DIR
wget -P $STRATEGY_QA_DIR https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip --no-check-certificate
unzip -d $STRATEGY_QA_DIR ${STRATEGY_QA_DIR}/strategyqa_dataset.zip
rm ${STRATEGY_QA_DIR}/strategyqa_dataset.zip

# Download ARC
ARC_DIR="${OUTPUT_DIR}"
wget -P $ARC_DIR https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip --no-check-certificate
unzip -d $ARC_DIR ${ARC_DIR}/ARC-V1-Feb2018.zip
rm ${ARC_DIR}/ARC-V1-Feb2018.zip

# Download CommonsenseQA
COMMONSENSE_QA_DIR="${OUTPUT_DIR}/commonsenseqa"
wget -P $COMMONSENSE_QA_DIR https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl --no-check-certificate
wget -P $COMMONSENSE_QA_DIR https://s3.amazonaws.com/commensenseqa/test_rand_split.jsonl --no-check-certificate