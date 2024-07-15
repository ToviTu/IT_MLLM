ARC 
1. Inference
For ARC dataset, go to llava/eval directory and run the following command:
```CUDA_VISIBLE_DEVICES=2 python model_ARC_loader.py --model-path /scratch/vision_share/models/llava-vicuna-7b-pretrain --model-base lmsys/vicuna-7b-v1.5```

# change the model-path to the path where the model is stored and the model-base to the one you want to evaluate

# the default is llava1.5, if you just run ```CUDA_VISIBLE_DEVICES=2 python model_ARC_loader.py```

# you can use --summarize-strategy llm to use llm to parse the input, but it is not recommended because the performance is not as good as the default parsing strategy - pattern

2. Evaluation
Go to llava/eval and run the following command
```CUDA_VISIBLE_DEVICES=2 python ARC_evaluator.py --ckpt vicuna```

The result will be stored at llava/eval/data/eval/arc

# The ARC inference for the second model will overwrite the first one, so make sure to move the first one to some other directory before running the second model like llava/eval/data/inference/vicuna

commonsenseqa
1. Inference
For commonsenseqa dataset, go to llava/eval directory and run the following command:
```CUDA_VISIBLE_DEVICES=2 python model_commonsense_qa_loader.py --model-path /scratch/vision_share/models/llava-vicuna-7b-pretrain --model-base lmsys/vicuna-7b-v1.5 ```

2. Evaluation
Go to llava/eval and run the following command
```CUDA_VISIBLE_DEVICES=3 python commonsense_evaluator.py --ckpt llava-vicuna-7b-pretrain```

cosmosqa
1. Inference
For cosmosqa dataset, go to llava/eval directory and run the following command:
```CUDA_VISIBLE_DEVICES=2 python model_cosmosqa_loader.py --model-path /scratch/vision_share/models/llava-vicuna-7b-pretrain --model-base lmsys/vicuna-7b-v1.5 ```

2. Evaluation
Go to llava/eval and run the following command
```CUDA_VISIBLE_DEVICES=0 python cosmosqa_evaluator.py --ckpt llava-vicuna-7b-pretrain```


#### for llama model 
#### example for ARC dataset
```CUDA_VISIBLE_DEVICES=3 python model_ARC_loader.py --model-path /scratch/vision_share/models/llava-llama2-7b-pretrain --model-base meta-llama/Llama-2-7b-hf```