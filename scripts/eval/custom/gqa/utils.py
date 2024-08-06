from datasets import load_dataset
import re
import json

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None

with open("/scratch/yu.zihao/llava-vicuna-7b-pretrain-fb_gqa/0730_0842_llava...in-fb_llava_plain_model_args_b1d452/gqa_custom_vicuna_pretrain.json", "r") as f:
    data = json.load(f)
    resps = [log['resps'] for log in data['logs']]

def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        GQA_RAW_IMAGE_DATASET = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]


def gqa_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    pattern = r'^(Is|Are|Am|Was|Were|Do|Does|Did|Has|Have|Had|Will|Would|Can|Could)\s'
    regex = re.compile(pattern, re.IGNORECASE) 
    if regex.match(question):
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
        post_prompt = model_specific_prompt_kwargs['post_prompt']
        prompt = f"{pre_prompt}{question}{post_prompt}"
    else:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
        post_prompt = model_specific_prompt_kwargs['post_prompt']
        prompt = f"{pre_prompt}{question}{post_prompt}"

    return prompt

def gqa_doc_to_text_parser(doc, model_specific_prompt_kwargs):
    target_id = doc['id']
    matching_resps = [[""]]
    for log in data['logs']:
        if log['doc']['id'] == target_id:
            matching_resps = log['resps']
            break

    question = doc["question"] + "\nLet's think step by step." + matching_resps[0][0]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs['post_prompt']
    prompt = f"{pre_prompt}{question}{post_prompt}"

    return prompt

def gqa_doc_to_text_finetuned_prompt(doc, model_specific_prompt_kwargs):

    question = doc["question"]
    pattern = r'^(Is|Are|Am|Was|Were|Do|Does|Did|Has|Have|Had|Will|Would|Can|Could)\s'
    regex = re.compile(pattern, re.IGNORECASE) 
    if regex.match(question):
        pre_prompt = ""
        post_prompt = "\nPlease answer the question with 'yes' or 'no'."
        prompt = f"{pre_prompt}{question}{post_prompt}"
    else:
        pre_prompt = ""
        post_prompt = "\nPlease provide a short answer to the question:"
        prompt = f"{pre_prompt}{question}{post_prompt}"

    return prompt