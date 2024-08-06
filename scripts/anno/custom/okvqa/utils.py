import random
import re
import os
import json

import datetime
import statistics

import lmms_eval.tasks._task_utils.file_utils as file_utils

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

from loguru import logger as eval_logger

from llava.conversation import conv_templates

with open(os.path.join(os.environ['WORKING_DIR'], "GPT_rationale/GPT_AOKVQA_rationale.json"), "r") as f:
        rationale = json.load(f)

def doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def process_results(doc, result):
    return {
        "exact_match": 0,
        "submission": {
            "question_id": doc["question_id"],
        },
    }

def doc_to_text(doc, model_specific_prompt_kwargs=None):
    # Load GPT rationale
    import random
    CoT_example = random.choice(rationale)

    # Load Template for LLaVA
    conv = conv_templates["v1"].copy()

    choices = doc['choices']
    #choices = re.findall(r"'(.*?)'", choices)
    choices_formatted = "Choose the best option from the following:\n" + "\n".join(choices)

    answer_labels = ["A", "B", "C", "D"]
    answer = answer_labels[int(doc['correct_choice_idx'])]

    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]
    
    # conv.append_message(conv.roles[0], f"{pre_prompt} Question: {CoT_example['question']} Answer: {CoT_example['answer']} {post_prompt}")
    # conv.append_message(conv.roles[1], CoT_example['rationale'])

    # conv.append_message(conv.roles[0], f"{pre_prompt} Question: {doc['question']} Answer: {random.choice(answers)} {post_prompt}")
    # conv.append_message(conv.roles[1], None)

    cot_prompt = f"{pre_prompt} Question: {CoT_example['question']} Answer: {CoT_example['answer']}{post_prompt} Rationale: {CoT_example['rationale']}</s>"
    prompt = f"{pre_prompt} <image>\nQuestion: {doc['question']} {choices_formatted} Answer: {answer}{post_prompt} Rationale:"
    return cot_prompt + prompt
    return conv.get_prompt()

def vqav2_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"vqav2-test-submission-{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")
