from llava.conversation import conv_templates
import string
import json
import os

def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))

def prepare_gt(input):
    return ['no', 'yes'][input['label']]

def prepare_input(input):
    question = input['question']+"?"
    context = input['passage']

    return context, question

def prompt_llama_plain(input):
    context, question = prepare_input(input)

    return f'Context: {context}\nQuestion: {question}\nAnswer: Among "yes" and "no", the best answer is "'

def prompt_llama_cot_extract(input):
    context, question = prepare_input(input)

    return f"Context: {context}\nQuestion: {question}\nAnswer: Let's think step by step."

COT_FILE = os.path.join(os.environ['STORAGE_DIR'], "results/test/meta-llama__Llama-2-7b-hf/samples_arc_easy_llama_cot_extract_2024-07-30T16-37-57.006009.jsonl")
if COT_FILE:
    rationale_map = {}
    with open(COT_FILE, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            rationale_map[entry['doc']['id']] = entry['arguments']['gen_args_0']['arg_0'] + entry['resps'][0][0]
def prompt_llama_cot_answer(input):
    assert COT_FILE, "COT_FILE is not set"
    assert input['id'] in rationale_map, f"ID {input['id']} not found in COT_FILE"
    return rationale_map[input['id']] + ' Among "yes" and "no", the best answer is'

def prompt_llava_cot(input):
    context, question = prepare_input(input)

    question_prompt = 'Context: '+context+'\nPlease answer with "yes" or "no":\n' + question

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()

def prompt_vicuna_plain(input):
    context, question = prepare_input(input)

    return f'User: Context:{context}\nPlease answer with "yes" or "no":\n{question}\nAssistant: The best answer is "'

def prompt_llava_plain(input):
    context, question = prepare_input(input)

    question_prompt = 'Please answer with "yes" or "no"\n'+question

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], "The best answer is \"")

    return conv.get_prompt()[:-4] # Manually remove </s> token