#! /bin/bash

. /scratch/peterni/wustl/IT_MLLM/sh/set_environ_var.sh
. ${WORKING_DIR}/sh/set_secrets.sh


echo $HF_HOME
singularity run --nv --bind /scratch,/storage1 /scratch/peterni/lang-modeling.sif /bin/bash

