#! /bin/bash

singularity run --nv --bind /scratch/ ./instruct-flamingo_latest.sif python ./Instruction-tuned-Flamingo-MLLM/evaluate_squad.py