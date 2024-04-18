#! /bin/bash

. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/src/set_environ_var.sh
singularity run --nv --bind /scratch,/storage1 /home/research/jianhong.t/instruct-flamingo_latest.sif /bin/bash
echo $HF_HOME
