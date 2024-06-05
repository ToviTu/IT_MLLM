#! /bin/bash

singularity run --nv --bind /scratch/ /scratch/t.tovi/instruct-flamingo_latest.sif \
    python /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/inference_yi_squad_cot.py
