#! /bin/bash

. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/sh/set_environ_var.sh
. ${WORKING_DIR}/sh/set_secrets.sh

singularity run --nv --bind /scratch/ /scratch/vision_share/eval.sif \
    bash /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/scripts/anno/lmm-eval.sh
