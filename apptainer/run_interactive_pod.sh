#! /bin/bash

. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/scripts/set_environ_var.sh
singularity run --nv --bind /scratch,/storage1 /scratch/t.tovi/lang-modeling.sif /bin/bash
echo $HF_HOME
