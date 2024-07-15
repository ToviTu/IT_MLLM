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