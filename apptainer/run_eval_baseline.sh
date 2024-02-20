#! /bin/bash

singularity run --nv --bind /scratch/ ./instruct-flamingo_latest.sif python ./Instruction-tuned-Flamingo-MLLM/eval_baseline.py