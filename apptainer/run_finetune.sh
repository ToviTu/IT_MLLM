#! /bin/bash

# Set environment variables
. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/sh/set_environ_var.sh
. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/sh/set_secrets.sh

echo $HF_HOME

# Execute script
singularity run --nv --bind /scratch,/storage1 /scratch/t.tovi/lang-modeling.sif \
    bash /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/scripts/train/finetune.sh
