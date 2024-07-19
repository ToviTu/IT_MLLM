#! /bin/bash

#export MODEL_DIR="/storage1/chenguangwang/Active/vision_share"
export STORAGE_DIR="/scratch/vision_share"
export WORKING_DIR="/home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM"
#export EVAL_DIR="/storage1/chenguangwang/Active/vision_share/dataset/eval"
#export PYTHONPATH="/scratch/peterni/wustl/IT_MLLM:$PYTHONPATH"

export HF_DATASETS_CACHE="${STORAGE_DIR}/datasets"
export HF_HOME="${STORAGE_DIR}/models"
#export TRANSFORMERS_CACHE="${STORAGE_DIR}/models"