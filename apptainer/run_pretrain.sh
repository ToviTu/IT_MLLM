#! /bin/bash

# Set environment variables
. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/scripts/set_environ_var.sh

# Execute script
singularity run --nv --bind /scratch,/storage1 /scratch/t.tovi/lang-modeling.sif \
    bash /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/scripts/train/pretrain.sh
echo $HF_HOME
