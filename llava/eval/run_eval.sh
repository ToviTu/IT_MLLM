#ARC
# code for inferencing
## llava-vicuna-7b-pretrain
CUDA_VISIBLE_DEVICES=2 python model_ARC_loader.py --model-path /scratch/vision_share/models/llava-vicuna-7b-pretrain --model-base lmsys/vicuna-7b-v1.5 

# code for evaluation
CUDA_VISIBLE_DEVICES=3 python ARC_evaluator.py --ckpt llava-vicuna-7b-pretrain 

## llava-llama2-7b-pretrain
CUDA_VISIBLE_DEVICES=3 python model_ARC_loader.py --model-path /scratch/vision_share/models/llava-llama2-7b-pretrain --model-base meta-llama/Llama-2-7b-hf

# code for evaluation
CUDA_VISIBLE_DEVICES=3 python ARC_evaluator.py --ckpt llava-llama2-7b-pretrain

# pure llama2
CUDA_VISIBLE_DEVICES=3 python model_ARC_loader.py --model-path meta-llama/Llama-2-7b-hf --model-base meta-llama/Llama-2-7b-hf

#commonsense_qa
# code for inferencing
## llava-vicuna-7b-pretrain
CUDA_VISIBLE_DEVICES=2 python model_commonsense_qa_loader.py --model-path /scratch/vision_share/models/llava-vicuna-7b-pretrain --model-base lmsys/vicuna-7b-v1.5 

# code for evaluation
CUDA_VISIBLE_DEVICES=3 python commonsense_evaluator.py --ckpt llava-vicuna-7b-pretrain

## llava-llama2-7b-pretrain
CUDA_VISIBLE_DEVICES=2 python model_commonsense_qa_loader.py --model-path /scratch/vision_share/models/llava-llama2-7b-pretrain --model-base meta-llama/Llama-2-7b-hf

# code for evaluation
CUDA_VISIBLE_DEVICES=3 python commonsense_evaluator.py --ckpt llava-llama2-7b-pretrain


#cosmosqa
# code for inferencing
## llava-vicuna-7b-pretrain
CUDA_VISIBLE_DEVICES=2 python model_cosmosqa_loader.py --model-path /scratch/vision_share/models/llava-vicuna-7b-pretrain --model-base lmsys/vicuna-7b-v1.5 

#code for evaluation
CUDA_VISIBLE_DEVICES=3 python cosmosqa_evaluator.py --ckpt llava-vicuna-7b-pretrain

## llava-llama2-7b-pretrain
CUDA_VISIBLE_DEVICES=1 python model_cosmosqa_loader.py --model-path /scratch/vision_share/models/llava-llama2-7b-pretrain --model-base meta-llama/Llama-2-7b-hf

#strategy_qa
# code for inferencing
## llava-vicuna-7b-pretrain
CUDA_VISIBLE_DEVICES=2 python model_strategy_qa_loader.py --model-path /scratch/vision_share/models/llava-vicuna-7b-pretrain --model-base lmsys/vicuna-7b-v1.5

## llava-llama2-7b-pretrain
CUDA_VISIBLE_DEVICES=2 python model_strategy_qa_loader.py --model-path /scratch/vision_share/models/llava-llama2-7b-pretrain --model-base meta-llama/Llama-2-7b-hf