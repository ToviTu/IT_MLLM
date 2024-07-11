#! /bin/bash

export MODEL_DIR="/storage1/chenguangwang/Active/vision_share"
export STORAGE_DIR="/scratch/peterni/wustl/"
export WORKING_DIR="/scratch/peterni/wustl/IT_MLLM/"
export EVAL_DIR="/storage1/chenguangwang/Active/vision_share/dataset/eval"
export PYTHONPATH="/scratch/peterni/wustl/IT_MLLM:$PYTHONPATH"

export HF_DATASETS_CACHE="${MODEL_DIR}/datasets"
export HF_HOME="${MODEL_DIR}/models"
export TRANSFORMERS_CACHE="${MODEL_DIR}/models"