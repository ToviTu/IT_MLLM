from llava.conversation import conv_templates
import string
import re
from lm_eval.api.filter import Filter
from datasets import load_dataset

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None

def doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        GQA_RAW_IMAGE_DATASET = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]
    
def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))

def prepare_gt(input):
    return input['answer']

def prompt_llama_plain(input, model_specific_prompt_kwargs=None):
    question = input['question']

    if "yes" in input['answer'] or "no" in input['answer']:
        return f'Question: {question}\nAnswer: Among "yes" and "no", the best answer is "'
    else: 
        return f'Question: Please provide a short answer to the question: {question}\nAnswer: The best answer is "'

# def prompt_llama_cot_extract(input, model_specific_prompt_kwargs=None):
#     question = prepare_input(input)

#     return f"Question: {question}\nAnswer: Let's think step by step."

# COT_FILE = ''
# if COT_FILE:
#     rationale_map = {}
#     with open(COT_FILE, 'r', encoding='utf-8') as file:
#         for line in file:
#             entry = json.loads(line)
#             rationale_map[entry['doc']['id']] = entry['arguments']['gen_args_0']['arg_0'] + entry['resps'][0][0]
# def prompt_llama_cot_answer(input, model_specific_prompt_kwargs=None):
#     assert COT_FILE, "COT_FILE is not set"
#     assert input['id'] in rationale_map, f"ID {input['id']} not found in COT_FILE"
#     return rationale_map[input['id']] + " Among A, B, C, and D, the best option is"

def prompt_llava_cot(input, model_specific_prompt_kwargs=None):
    question = input['question']

    if "yes" or "no" in input['answer']:
        question_prompt = 'Please answer the question with "yes" or "no".' + question + "\n"
    else:
        question_prompt = "Please provide a short answer to the question:" + question + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()

def prompt_vicuna_plain(input, model_specific_prompt_kwargs=None):
    question = input['question']

    if "yes" or "no" in input['answer']:
        return f'User: {question} Please answer the question with "yes" or "no".\nAssistant: The best answer is "'
    else: 
        return f'User: {question}\nAssistant: The best answer is "'

def prompt_llava_plain(input, model_specific_prompt_kwargs=None):
    question = input['question']

    if "yes" in input['answer'].lower() or "no" in input['answer'].lower():
        question_prompt = 'Please answer the question with "yes" or "no". ' + question + "\n"
    else:
        question_prompt = "Please provide a short answer to the question: " + question + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt() + ' The best answer is "' # Manually remove </s> token
    
class Digit2NumberFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst, doc):
            mapper = {
                "none": "0",
                "zero": "0",
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9"
            }

            mapped_resps = []
            for raw_resp in inst:
                resp = raw_resp.lower().strip()
                if resp in mapper:
                    mapped_resps.append(mapper[resp])
                else:
                    mapped_resps.append(raw_resp)
            return mapped_resps

        filtered_resps = list(map(lambda x: filter_set(x[0], x[1]), zip(resps, docs)))
        return filtered_resps
