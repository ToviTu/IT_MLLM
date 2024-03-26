#! /bin/bash
export HF_DATASETS_CACHE="/scratch/t.tovi/datasets/"
export HF_HOME="/scratch/t.tovi/models/"
export TRANSFORMERS_CACHE="/scratch/t.tovi/models/"
singularity run --nv --bind /scratch/ ./instruct-flamingo_latest.sif python ./Instruction-tuned-Flamingo-MLLM/evaluate_GSM8K.py