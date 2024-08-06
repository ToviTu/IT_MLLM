from llava.conversation import conv_templates
import string
from lm_eval.api.filter import Filter


def aokvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]
    
def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))

def prepare_gt(input):
    return input['answer']

def prepare_input(input):
    question = input['question']

    return question

def prompt_llama_plain_logit(input, model_specific_prompt_kwargs=None):
    question = prepare_input(input)

    return f'Question: {question}\nAnswer:'

def prompt_llama_plain(input, model_specific_prompt_kwargs=None):
    question = prepare_input(input)

    return f'Question: {question}\nAnswer: Among "yes" and "no", the best answer is "'

# def prompt_llama_cot_extract(input, model_specific_prompt_kwargs=None):
#     question, choices_formatted = prepare_input(input)

#     return f"Question: {question} Choose the best option from below:\n{choices_formatted}\nAnswer: Let's think step by step."

# COT_FILE = os.path.join(os.environ['STORAGE_DIR'], "results/test/meta-llama__Llama-2-7b-hf/samples_arc_easy_llama_cot_extract_2024-07-30T16-37-57.006009.jsonl")
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
    question = prepare_input(input)

    question_prompt = 'Please answer with "yes" or "no":' + "\n" + question + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()

def prompt_vicuna_plain(input, model_specific_prompt_kwargs=None):
    question = prepare_input(input)

    return f'User: Please answer with "yes" or "no":\n{question}\nAssistant: The best answer is "'

def prompt_llava_plain(input, model_specific_prompt_kwargs=None):
    question = prepare_input(input)

    question_prompt = 'Please answer with "yes" or "no":' + "\n" + question + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt() + ' The best answer is "' # Manually remove </s> token

