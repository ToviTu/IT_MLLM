from datasets import load_dataset, load_metric
from llava.anno.prompt_template import llava_cqa, llava_cmc, fa_prompt, llava_srq, llava_tfq
import json
import numpy as np
import csv
import string
import re
import os
import tqdm
import random

DATA_DIR = os.environ["HF_DATASETS_CACHE"]

bool_mapper = lambda answer: "yes" if answer else "no"

class DatasetFactory:

    def __init__(self, train_set, val_set):
        self.training = True
        self.train_set = train_set
        self.val_set = val_set

        assert "question" in self.train_set[0] and "answer" in self.train_set[0], "Expect question and answer columns in the dataset"

    def format_input(self, context="", question="", choices="", answer="", template=llava_srq, prefix='', suffix=''):
        '''
        Format the input for the model
        '''
        final_answer = answer.split(" ")[-1]

        if choices:
            formatted_input = prefix + llava_cmc(context, choices, question, answer) + suffix
        elif "yes" in final_answer or "no" in final_answer:
            formatted_input = prefix + llava_tfq(context, question, answer) + suffix
        else:
            formatted_input = prefix + template(context, question, answer) + suffix
        return formatted_input
    
    def add_rationale(self, rationale):
        self.rationale = rationale
    
    def format_input_with_cot_example(self, eos_token, instruction, cot_prompts, cqa_template, cqa):

        # Randomly select a cot exmaple
        cot_example = cot_prompts[np.random.choice(len(cot_prompts))]
        cot_example = self.format_input(
                context = cot_example['context'] if 'context' in cot_example else '',
                question = cot_example['question'],
                choices= cot_example['choices'] if 'choices' in cot_example else '',
                answer = cot_example['answer'],
                template = cqa_template, 
                prefix = instruction + '\n', 
                suffix = '\n' + "Rationale: " + cot_example['Rationale'] + eos_token
            )

        # Apply template to the question
        question = self.format_input(
                context = cqa['context'] if 'context' in cqa else '',
                question = cqa['question'],
                answer = cqa['answer'], 
                template = cqa_template,
                prefix = instruction + '\n',
                suffix = '\n' + "Rationale:" 
            )

        return cot_example + question

    def process(self, split='train', mode='train', template=llava_cqa, eos_token='', instruction='', cot=False, cot_prompts=None):
        '''
        Format the datasets accoridng to the needs
        '''
        data = self.train_set if split == 'train' else self.val_set
        if cot:
            assert cot_prompts is not None, "CoT prompts are required"
            mapping_func = lambda example: {
                    'id': example['id'],
                    'finputs': self.format_input_with_cot_example(eos_token, instruction, cot_prompts, template, example),
                }
        else:
            mapping_func = lambda example: {
                    'id': example['id'],
                    'finputs': self.format_input(
                        context=example['context'] if 'context' in example else "", 
                        question=example['question'], 
                        choices=example['choices'] if "choices" in example else "",
                        answer=example['answer'] if mode=='train' else "", 
                        template=template,
                        prefix=instruction + '\n' if instruction else "",
                    ),
                }
            
        return map(mapping_func, data)
        
    def process_with_rationale(self, rationale=None, template=llava_srq, instruction=''):
        mapping_func = lambda example, rationale: {
                'id': example['id'],
                'finputs': self.format_input(
                    context=example['context'] if 'context' in example else "", 
                    question=example['question'], 
                    choices=example['choices'] if "choices" in example else "",
                    answer=rationale + f"\n{random.choice(fa_prompt)} " + example['answer'] + ".", 
                    template=template,
                    prefix=instruction + '\n' if instruction else "",
                ),
            }

        data = []
        for example in tqdm.tqdm(self.train_set):
            if example['id'] not in rationale:
                continue
            r = rationale[example['id']]
            data.append(mapping_func(example, r))
        return data


class SQuAD(DatasetFactory):

    def __init__(self):
        data = load_dataset("squad")

        train_set = data['train']
        val_set = data['validation']

        preprocess = lambda instance: {
            'id': instance['id'],
            'context': instance['context'],
            'question': instance['question'],
            'answer': instance['answers']['text'][0],
        }
        
        train_set = list(map(preprocess, train_set))
        val_set = list(map(preprocess, val_set))

        super().__init__(train_set, val_set)

        self.metric = load_metric("squad")
    
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def extract_answer(self, model_answers):
        '''
        Extract the succinct answers from the model generated texts
        '''
        all_answers = {entry['id']: entry['answers'] for entry in self.val_set} #all possible answers
        
        extracted_answers = []
        ground_truth = []
        for entry in model_answers:
            id = entry['id']
            text = entry['prediction_text']
            possible_answers = all_answers[id]['text']

            # Preprocess predictions
            text = self.normalize_answer(text)

            # Loop through all answers
            is_matched = False
            for each in possible_answers:
                each = self.normalize_answer(each)
                # If the exact answer is mentioned
                if text.find(each) != -1:
                    extracted_answers.append({'id': id, 'prediction_text': each})
                    is_matched = True
                    break
            # No match
            if not is_matched:
                extracted_answers.append({'id': id, 'prediction_text': text})
            ground_truth.append({'id':id, 'answers': all_answers[id]})
        return extracted_answers, ground_truth, self.metric.compute(predictions=extracted_answers, references=ground_truth)

class StrategyQA(DatasetFactory):

    def __init__(self):
        with open(f"{DATA_DIR}/strategyqa/strategyqa_train.json", 'r') as f:
            data = json.load(f)
        train_set = data


        preprocess = lambda instance: {
            'id': instance['qid'],
            'question': instance['question'],
            'context': " ".join(instance['facts']),
            'answer': bool_mapper(instance['answer']),
        }
        train_set = list(map(preprocess, train_set))
        super().__init__(train_set, [])


    def extract_answer(self, model_answers):
        '''
        Extract the succinct answers from the model generated texts
        '''
        all_answers = {entry['qid']: entry['answers'] for entry in self.val_set} #all possible answers

        extracted_answers = []
        ground_truth = []
        for entry in model_answers:
            id = entry['id']
            text = entry['prediction_text']
            possible_answers = all_answers[id]['text']

            is_matched = False
            for each in possible_answers:
                if text.find(each) != -1:
                    extracted_answers.append({'id': id, 'prediction_text': each})
                    is_matched = True
                    break
            if not is_matched:
                extracted_answers.append({'id': id, 'prediction_text': text})
            ground_truth.append({'id':id, 'answers': all_answers[id]})
        return extracted_answers, ground_truth

class CommonsenseQA(DatasetFactory):

    def __init__(self):
        # Read raw dataset
        data = []
        with open(f"{DATA_DIR}/commonsenseqa/train_rand_split.jsonl", 'r') as file:
            for line in file:
                data.append(json.loads(line))

        # Format question inputs
        fdata = []
        for entry in data:
            fchoices = []
            for choice in entry['question']['choices']:
                fchoices.append(choice['label'] + " " + choice['text'])
            
            fentry = {}
            fentry['id'] = entry['id']
            fentry['question'] = entry['question']['stem']
            fentry['choices'] = fchoices
            fentry['answer'] = entry['answerKey']

            fdata.append(fentry)

        super().__init__(fdata, [])

class CosmosQA(DatasetFactory):

    def __init__(self):
        # Read raw dataset
        data = []
        with open(f"{DATA_DIR}/cosmosqa/train.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:
                id = line[0]
                context = line[1]
                question = line[2]
                choices = [
                    "A " + line[3],
                    "B " + line[4],
                    "C " + line[5],
                    "D " + line[6]
                ]
                mapping = {"0":"A", "1":"B", "2":"C", "3":"D"}
                answer = mapping[line[-1]]
                entry = {"id": id, "context": context, "question": question, "choices": choices, "answer": answer}
                data.append(entry)

        super().__init__(data, [])

class ARC(DatasetFactory):

    def __init__(self):
        # Read raw dataset
        data = []
        with open(f"{DATA_DIR}/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl", 'r') as f:
            for line in f:
                data.append(json.loads(line))

        with open(f"{DATA_DIR}/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl", 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Format choices
        train_set = []
        for entry in data:
            fentry = {}
            fentry['id'] = entry['id']
            fentry['question'] = entry['question']['stem'] + "..." if entry['question']['stem'][-1] != '?' else entry['question']['stem']
            choices = [choice['label']+ " " + choice['text'] for choice in entry['question']['choices']]
            fentry['choices'] = choices
            fentry['answer'] = entry['answerKey']
            train_set.append(fentry)
        
        super().__init__(train_set, [])

class VQAv2(DatasetFactory):
    
    def __init__(self):
        data = load_dataset("Multimodal-Fatima/VQAv2_train", split="train[:20000]")
        train_set = data

        preprocess = lambda instance: {
            'id': str(instance['question_id']),
            'image': instance['image'],
            'image_id': instance['id_image'],
            'question': instance['question'],
            'answer': instance['multiple_choice_answer'],
        }
        
        train_set = list(map(preprocess, train_set))
        val_set = []

        super().__init__(train_set, val_set)
    
    def process_with_rationale(self, rationale=None, template=llava_srq, instruction='', output_dir=None):
        mapping_func = lambda example, rationale: {
                'id': example['id'],
                'finputs': self.format_input(
                    context=example['context'] if 'context' in example else "", 
                    question="<image>\n"+example['question'], 
                    choices=example['choices'] if "choices" in example else "",
                    answer=rationale + f"\n{random.choice(fa_prompt)} " + example['answer'] + ".", 
                    template=template,
                    prefix=instruction + '\n' if instruction else "",
                ),
                "image": str(example['image_id'])+".jpg",
            }

        data = []
        for example in tqdm.tqdm(self.train_set):
            if example['id'] not in rationale:
                continue
            r = rationale[example['id']]
            data.append(mapping_func(example, r))
            if output_dir:
                image = example['image'].convert("RGB")
                image.save(f"{output_dir}/{example['image_id']}.jpg")

        return data

class AOKvqa(DatasetFactory):
    
    def __init__(self):
        data = load_dataset("HuggingFaceM4/A-OKVQA", split="train[:20000]")
        train_set = data

        mc_mapper = ['A', 'B', 'C', 'D', 'E']

        preprocess = lambda instance: {
            'id': instance['question_id'],
            'image': instance['image'],
            'image_id': instance['question_id'],
            'question': instance['question'],
            'choices': [f"{mc_mapper[i]} {choice}" for i, choice in enumerate(instance['choices'])],
            'answer': mc_mapper[int(instance['correct_choice_idx'])],
        }
        
        train_set = list(map(preprocess, train_set))
        val_set = []

        super().__init__(train_set, val_set)
    
    def process_with_rationale(self, rationale=None, template=llava_srq, instruction='', output_dir=None):
        mapping_func = lambda example, rationale: {
                'id': example['id'],
                'finputs': self.format_input(
                    context=example['context'] if 'context' in example else "", 
                    question="<image>\n"+example['question'], 
                    choices=example['choices'] if "choices" in example else "",
                    answer=rationale + f"\n{random.choice(fa_prompt)} " + example['answer'] + ".", 
                    template=template,
                    prefix=instruction + '\n' if instruction else "",
                ),
                "image": str(example['image_id'])+".jpg",
            }

        data = []
        for example in tqdm.tqdm(self.train_set):
            if example['id'] not in rationale:
                continue
            r = rationale[example['id']]
            data.append(mapping_func(example, r))
            if output_dir:
                image = example['image'].convert("RGB")
                image.save(f"{output_dir}/{example['image_id']}.jpg")

        return data