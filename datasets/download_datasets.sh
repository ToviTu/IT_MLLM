#!/bin/bash

set -e

WORKING_DIR=${WORKING_DIR:-/default/working/path}
echo "Working directory: $WORKING_DIR"

DEST_DIR="$WORKING_DIR/datasets/"

mkdir -p "$DEST_DIR"

cd "$DEST_DIR"

# ARC dataset
echo "Downloading ARC dataset..."
wget --no-check-certificate https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip -O ARC-V1-Feb2018.zip

echo "Unzipping ARC dataset..."
unzip -o ARC-V1-Feb2018.zip

rm ARC-V1-Feb2018.zip

echo "ARC dataset downloaded and unzipped successfully."

# CommonsenseQA dataset
COMMONSENSEQA_DIR="$WORKING_DIR/datasets/commonsenseqa"

mkdir -p "$COMMONSENSEQA_DIR"

cd "$COMMONSENSEQA_DIR"

echo "Downloading CommonsenseQA dataset..."
wget --no-check-certificate https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl -O test_rand_split.jsonl

echo "CommonsenseQA dataset downloaded successfully."

# StrategyQA dataset
STRATEGYQA_DIR="$WORKING_DIR/datasets/strategyqa"

mkdir -p "$STRATEGYQA_DIR"

cd "$STRATEGYQA_DIR"

echo "Downloading StrategyQA dataset..."
wget --no-check-certificate https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip -O strategyqa_dataset.zip

echo "Unzipping StrategyQA dataset..."
unzip strategyqa_dataset.zip

rm strategyqa_dataset.zip

echo "StrategyQA dataset downloaded and unzipped successfully."

# CosmosQA dataset
COSMOSQA_DIR="$WORKING_DIR/datasets/cosmosqa"

mkdir -p "$COSMOSQA_DIR"

cd "$COSMOSQA_DIR"

echo "Downloading CosmosQA dataset..."
wget --no-check-certificate https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/train.csv -O train.csv
wget --no-check-certificate https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/valid.csv -O dev.csv
wget --no-check-certificate https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/test.jsonl -O test.jsonl

echo "CosmosQA dataset downloaded successfully."