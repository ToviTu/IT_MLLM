from llava.conversation import conv_templates
import string
import re
from lm_eval.api.filter import Filter

MC_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."
SA_PROMPT = "Please answer directly with a single word or number."
DIGIT_PROMPT = "Please answer directly with a single word or number."

def doc_to_visual(doc):
    return [doc["image"].convert("RGB")]
    
def remove_punctuation(input_string):
    return input_string.translate(str.maketrans('', '', string.punctuation))

def parse_question(input):
    question = input['question']
    question = question.replace(MC_PROMPT, "").replace(SA_PROMPT, "").replace(DIGIT_PROMPT, "")
    choices = re.findall("[ABC]\.\s(.+)\s", question)
    q = question.split('\n')[0]
    if q.strip()[-1] not in string.punctuation:
        q = q + '...'
    if choices:
        return {'question': q, 'choices': list(choices)}
    else:
        return {'question': q}

def prepare_gt(input):
    return input['answer']

def prepare_input(input):
    question_dict = parse_question(input)

    question = question_dict['question']
    choices = question_dict.get('choices', "")
    labels = ['A', 'B', 'C']

    if choices:
        choices_w_labels = [f"{label}: {choice}" for label, choice in zip(labels, choices)]
        choices_formatted = "\n".join(choices_w_labels)
    else:
        choices_formatted = ""

    return question, choices_formatted

def prompt_llama_plain(input, model_specific_prompt_kwargs=None):
    question, choices_formatted = prepare_input(input)

    if choices_formatted:
        return f'Question: {question} Choose the best option from below:\n{choices_formatted}\nAnswer: The best answer is "'
    elif "yes" in input['answer'] or "no" in input['answer']:
        return f'Question: {question} Please answer the question with "yes" or "no".\nAnswer: The best answer is "'
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
    question, choices_formatted = prepare_input(input)

    if choices_formatted:
        question_prompt = question + " Choose the best option from below:\n" + choices_formatted + "\n"
    elif "yes" in input['answer'] or "no" in input['answer']:
        question_prompt = 'Please answer the question with "yes" or "no".' + question + "\n"
    else:
        question_prompt = "Please provide a short answer to the question:" + question + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()

def prompt_vicuna_plain(input, model_specific_prompt_kwargs=None):
    question, choices_formatted = prepare_input(input)

    if choices_formatted:
        return f'User: {question} Choose the best option from the choices provided:\n{choices_formatted}\nAssistant: The best answer is "'
    elif "yes" or "no" in input['answer']:
        return f'User: {question} Please answer the question with "yes" or "no".\nAssistant: The best answer is "'
    else: 
        return f'User: {question}\nAssistant: The best answer is "'

def prompt_llava_plain(input, model_specific_prompt_kwargs=None):
    question, choices_formatted = prepare_input(input)

    if choices_formatted:
        question_prompt = question + " Choose the best option from below:\n" + choices_formatted + "\n"
    elif "yes" in input['answer'].lower() or "no" in input['answer'].lower():
        question_prompt = 'Please answer the question with "yes" or "no". ' + question + "\n"
    else:
        question_prompt = "Please provide a short answer to the question: " + question + "\n"

    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], question_prompt)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt() + ' The best answer is "' # Manually remove </s> token
    
class AnswerMappingFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst, doc):
            question_dict = parse_question(doc)

            question = question_dict['question']
            choices = question_dict.get('choices', "")
            choices = [
                each.lower().strip().translate(str.maketrans('', '', string.punctuation))
                for each in choices
            ]

            mapped_resps = []
            for raw_resp in inst:
                resp = raw_resp.lower().strip()

                if resp in ['A', 'B', 'C', 'D']:
                    mapped_resps.append(resp)
                elif resp in choices:
                    index = choices.index(resp)
                    mapped_resps.append(['A', 'B', 'C'][index])
                else:
                    mapped_resps.append(raw_resp) 
            return mapped_resps

        filtered_resps = []
        for resp, doc in zip(resps, docs):
            filtered_resps.append(filter_set(resp, doc))

        #flat_filtered_resps = [item for sublist in filtered_resps for item in sublist] if any(isinstance(i, list) for i in filtered_resps) else filtered_resps

        return filtered_resps

class Digit2NumberFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst, doc):
            question_dict = parse_question(doc)

            question = question_dict['question']
            choices = question_dict.get('choices', "")
            choices = [each.lower() for each in choices]

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
