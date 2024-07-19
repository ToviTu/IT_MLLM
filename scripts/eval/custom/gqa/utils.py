from datasets import load_dataset
import re

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None


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
    # question = doc["question"]
    # pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    # post_prompt = model_specific_prompt_kwargs["post_prompt"]
    question = doc["question"]
    pattern = r'^(Is|Are|Am|Was|Were|Do|Does|Did|Has|Have|Had|Will|Would|Can|Could)\s'
    regex = re.compile(pattern, re.IGNORECASE) 
    if regex.match(question):
        pre_prompt = ""
        post_prompt = "\nAnswer the question using Yes or No."
        prompt = f"{pre_prompt}{question}{post_prompt}"
    else:
        pre_prompt = ""
        post_prompt = "\nAnswer the question using a single word or phrase."
        prompt = f"{pre_prompt}{question}{post_prompt}"

    return prompt
