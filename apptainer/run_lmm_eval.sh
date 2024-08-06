#! /bin/bash

# Set environment variables
. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/sh/set_environ_var.sh
. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/sh/set_secrets.sh

echo $HF_HOME

# Execute script
singularity run --nv --bind /scratch,/storage1 /scratch/vision_share/eval.sif \
    bash /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/scripts/eval/lmm-eval.sh
