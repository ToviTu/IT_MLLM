import re
import os
import json
import yaml
import pathlib

import datetime
import statistics

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
from lmms_eval.filters.extraction import RegexFilter

from loguru import logger as eval_logger
from llava.anno.prompt_template import llava_cmc


def aokvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def multiple_choice_prompt(doc):
    question = doc["question"]
    choices = doc["choices"]
    

def aokvqa_process_results(doc, result):

    resAns = result[0]
    accuracy = 0

    if "direct_answers" in doc and doc["direct_answers"] is not None:
        gtAcc = []

        for i in range(len(doc["direct_answers"])):
            doc["direct_answers"][i] = eval_ai_processor(doc["direct_answers"][i])

        for i in range(len(doc["direct_answers"])):
            otherGTAns = [doc["direct_answers"][j] for j in range(len(doc["direct_answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        if gtAcc:
            accuracy = statistics.mean(gtAcc)
        else:
            accuracy = 0

    return {
        "exact_match": accuracy,
        "submission": {
            "image": f"{doc['question_id']}.jpg",
            "answer": resAns,
        },
    }


def aokvqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    question = doc["question"]
    question += "\n" + f"A. {doc['choices'][0]}\n"
    question += f"B. {doc['choices'][1]}\n"
    question += f"C. {doc['choices'][2]}\n"
    question += f"D. {doc['choices'][3]}"
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def aokvqa_doc_to_text_custom_prompt(doc, model_specific_prompt_kwargs=None):
    question = doc["question"] + "Choose the best option from the choices provided:"
    question += "\n" + f"A. {doc['choices'][0]}\n"
    question += f"B. {doc['choices'][1]}\n"
    question += f"C. {doc['choices'][2]}\n"
    question += f"D. {doc['choices'][3]}"
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in model_specific_prompt_kwargs:
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if "post_prompt" in model_specific_prompt_kwargs:
        post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def aokvqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m%d-%H%M-%S")
    file = f"aokvqa-test-submission-{now_date_time}.json"
    path = generate_submission_file(file, args)
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Submission file saved to {path}")

def aokvqa_doc_to_target(doc):
    return ast.literal_eval(doc["direct_answers"])