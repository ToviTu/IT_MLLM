#! /bin/bash

python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava_plain \
    --model_args pretrained="liuhaotian/llava-v1.6-34b" \
    --include_path ${WORKING_DIR}/scripts/anno/custom/ \
    --tasks aokvqa_anno \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix anno \
    --output_path /scratch/vision_share/results/anno \
    --verbosity DEBUG

# python3 -m lmms_eval \
#     --model llava_plain \
#     --model_args pretrained="liuhaotian/llava-v1.6-34b",device_map=auto\
#     --include_path ${WORKING_DIR}/scripts/anno/custom/ \
#     --tasks aokvqa_anno \
#     --batch_size 1 \
#     --log_samples \
#     --gen_kwargs do_sample=False \
#     --log_samples_suffix anno \
#     --limit 20 \
#     --output_path /scratch/vision_share/results/test \
#     --verbosity DEBUG