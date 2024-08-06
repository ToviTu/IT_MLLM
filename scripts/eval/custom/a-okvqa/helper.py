from llava.conversation import conv_templates
import string
import json
import os
import statistics
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


def aokvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]
    
def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))

def prepare_gt(input):
    return ['A', 'B', 'C', 'D'][input['correct_choice_idx']]

def prepare_input(input):
    question = input['question']
    choices_text = input['choices']
    choices_label = ['A', 'B', 'C', 'D']
    choices_formatted = "\n".join([f"{label}: {text}" for label, text in zip(choices_label, choices_text)])

    return question, choices_formatted

def prompt_llama_plain_logit(input, model_specific_prompt_kwargs=None):
    question, choices_formatted = prepare_input(input)

    return f'Question: {question}\nAnswer:'

def prompt_llama_plain(input, model_specific_prompt_kwargs=None):
    question, choices_formatted = prepare_input(input)

    return f'Question: {question} Choose the best option from below:\n{choices_formatted}\nAnswer: The best answer is "'

def prompt_llama_cot_extract(input, model_specific_prompt_kwargs=None):
    question, choices_formatted = prepare_input(input)

    return f"Question: {question} Choose the best option from below:\n{choices_formatted}\nAnswer: Let's think step by step."

COT_FILE = os.path.join(os.environ['STORAGE_DIR'], "results/test/meta-llama__Llama-2-7b-hf/samples_arc_easy_llama_cot_extract_2024-07-30T16-37-57.006009.jsonl")
if COT_FILE:
    rationale_map = {}
    with open(COT_FILE, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            rationale_map[entry['doc']['id']] = entry['arguments']['gen_args_0']['arg_0'] + entry['resps'][0][0]
def prompt_llama_cot_answer(input, model_specific_prompt_kwargs=None):
    assert COT_FILE, "COT_FILE is not set"
    assert input['id'] in rationale_map, f"ID {input['id']} not found in COT_FILE"
    return rationale_map[input['id']] + " Among A, B, C, and D, the best option is"

def prompt_llava_cot(input, model_specific_prompt_kwargs=None):
    question, choices_formatted = prepare_input(input)

    question_prompt ="<image>" + question + " Choose the best option from below:\n" + choices_formatted + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()

def prompt_vicuna_plain(input, model_specific_prompt_kwargs=None):
    question, choices_formatted = prepare_input(input)

    return f'User: {question} Choose the best option from the choices provided:\n{choices_formatted}\nAssistant: The best answer is "'

def prompt_llava_plain(input, model_specific_prompt_kwargs=None):
    question, choices_formatted = prepare_input(input)

    question_prompt = question + " Choose the best option from below:\n" + choices_formatted + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], "The answer is")

    return conv.get_prompt()[:-4] # Manually remove </s> token

def answer_mapping(doc, result):
    pred = result[0]

    if pred in ["A", "B", "C", "D"]:
        acc = 1 if pred.lower() == prepare_gt(doc).lower() else 0
    else:
        acc = 1 if pred.lower() == doc['choices'][doc['correct_choice_idx']].lower() else 0
    
    return {
        "exact_match": acc
    }

def aokvqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
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
    }
