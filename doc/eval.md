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

# Visual Tasks guideline

All the bash files for running inference are located at: `IT_MLLM/scripts/v1_5/eval`. 
Each bash file includes three sections: 1. Prepare datasets 2. inference, and 3. parse and evaluate. 

### 1. Prepare datasets
Vizwiz and GQA has question file provides by LLaVA repo. The code provides customize usage for
modifying prompt. 
A-OKVQA is downloaded from A-OKVQA website. The question file is reformated for inference. 

### 2. Inference
The inference is achieved with `llava.eval.model_vqa_loader` provided by LLaVA Repo. 
`llava.eval.model_vqa_loader` can be adapted to work with any VQA datasets by simply changing the corresponding 
keys to "question_id", "image", and "text" in the question file of VQA datasets. 

### 3. Parse and Evaluate
Vizwiz is parsed and submitted to Eval AI for evaluation.
GQA and A-OKVQA are parsed and evaluated locally. 

### Vizwiz
```bash
CUDA_VISIBLE_DEVICES=0 bash vizwiz_llama_finetuned.sh
CUDA_VISIBLE_DEVICES=1 bash vizwiz_vicuna_finetuned.sh
```
### GQA
GQA test set need to be uploaded to the [leaderboard](https://leaderboard.allenai.org/a-okvqa/submissions/get-started) for evaluation. 
GQA validation set is tested in this experiment and score is computed locally at `1_eval.py`. 
GQA supports multiple GPU inference. 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash gqa_llama_finetuned.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash gqa_vicuna_finetuned.sh
```
### A-OKVQA
```bash
CUDA_VISIBLE_DEVICES=0 bash a-okvqa_llama_finetuned.sh
CUDA_VISIBLE_DEVICES=1 bash a-okvqa_vicuna_finetuned.sh
```