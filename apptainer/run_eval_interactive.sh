#! /bin/bash

. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/sh/set_environ_var.sh
. ${WORKING_DIR}/sh/set_secrets.sh

echo $HF_HOME
singularity run --nv --bind /scratch,/storage1 /scratch/vision_share/eval.sif /bin/bash

