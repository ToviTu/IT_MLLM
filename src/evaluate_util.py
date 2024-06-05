from datasets import load_dataset, load_metric
from datasets import Dataset as HDataset
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import csv
import pandas as pd
import string
import re

DATA_DIR = "/scratch/t.tovi/dataset/"

class SQuAD:

    def __init__(self):
        data = load_dataset("squad")
        self.train_set = data['train']
        self.val_set = data['validation']
        self.metric = load_metric("squad")
    
    def format_input_with_cot_prompt(self, eos_token, instruction, cot_prompts, cqa_template, cqa):

        # Prepend instruction
        formatted_input = instruction + '\n'

        # Randomly select a cot exmaple
        cot_example = cot_prompts[np.random.choice(len(cot_prompts))]
        cot_template = cqa_template(cot_example['Context'], cot_example['Question'], cot_example['Answer'])
        formatted_input += cot_template+'\n'

        # Append the gpt rationale
        formatted_input += "Rationale: " + cot_example['Rationale'] + eos_token

        # Apply template to the question
        inputs = cqa_template(cqa['context'], cqa['question'], cqa['answers']['text'][0])
        formatted_input += instruction + '\n' + inputs + '\n'

        # Add Rationale prompt
        formatted_input += "Rationale:"

        return formatted_input
    
    def get_train_set(self, eos_token, instruction, template, cot_prompts, batchsize=32):
        '''
        Select only context and question columns
        '''
        preprocessed_dataset = self.train_set.map(
            lambda example: 
                {
                    'finputs': self.format_input_with_cot_prompt(eos_token, instruction, cot_prompts, template, example),
                }, 
                remove_columns=[c for c in self.train_set.column_names if c != 'id']
        )
        
        # Convert encoded batched examples into tensors
        preprocessed_dataset.set_format('torch')
        return preprocessed_dataset
    
    def get_eval_set(self, cqa_template):
        '''
        Select only context and question columns
        '''
        preprocessed_dataset = self.val_set.map(
            lambda example: 
                {
                    'finputs': cqa_template(example['context'], example['question'], ""),
                }, 
                remove_columns=[c for c in self.train_set.column_names if c != 'id']
        )
        
        # Convert encoded batched examples into tensors
        preprocessed_dataset.set_format('torch')
        return preprocessed_dataset
    
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

class StrategyQA:

    def __init__(self):
        with open("/scratch/t.tovi/dataset/strategyqa_train.json", 'r') as f:
            data = json.load(f)
        self.train_set = data

    def format_input_with_cot_prompt(self, eos_token, instruction, cot_prompts, cqa_template, cqa):

        # Prepend instruction
        formatted_input = instruction + '\n'

        # Randomly select a cot exmaple
        cot_example = cot_prompts[np.random.choice(len(cot_prompts))]
        cot_template = cqa_template(cot_example['facts'], cot_example['question'], cot_example['answer'])
        formatted_input += cot_template+'\n'

        # Append the gpt rationale
        formatted_input += "Rationale: " + cot_example['rationale'] + eos_token

        # Apply template to the question
        facts = " ".join(cqa['facts'])
        inputs = cqa_template(facts, cqa['question'], cqa['answer'])
        formatted_input += instruction +  '\n' + inputs+'\n'

        # Add Rationale prompt
        formatted_input += "Rationale:"

        return formatted_input

    def get_train_set(self, instruction, eos_token, template, cot_prompts, batchsize=32):
        '''
        Select only context and question columns
        '''
        preprocessed_dataset = []
        for batch in self.train_set:
            fbatch = {}
            fbatch['id'] = batch['qid']

            fbatch['finputs'] = self.format_input_with_cot_prompt(eos_token, instruction, cot_prompts, template, batch)
            preprocessed_dataset.append(fbatch)

        return preprocessed_dataset

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

class GSM8K:

    def __init__(self):
        data = load_dataset('gsm8k', 'main')
        self.train_set = data['train']
        self.val_set = data['test']

        # Reserve 10% of the training set for few-shot prompting
        idx = np.random.choice(range(len(self.train_set)),  int(0.1 * len(self.train_set)))
        self.prompt_set = self.train_set[idx]
        train_idx = list(set(range(len(self.train_set))) - set(idx))
        self.train_set = self.train_set[train_idx]

    def get_eval_set(self, processor, template, batchsize=32):

        # Select only context and question columns
        preprocessed_dataset = self.val_set.map(
            lambda example: 
                {
                    'finputs': template(example['question']),
                }, 
                remove_columns=self.val_set.column_names
        )

        # Get max length
        max_len = 256

        # Tokenize the inputs
        def tokenize_function(examples):
            encoded = processor(
                    examples['finputs'],
                    padding='max_length',
                    max_length=max_len,
                )

            return {k: encoded[k] for k in encoded.keys() if k != "pixel_values"} 
                    
        preprocessed_dataset = preprocessed_dataset.map(
            tokenize_function, 
            remove_columns=[c for c in preprocessed_dataset.column_names if c != 'id'],
            batched = True,
            batch_size = batchsize
        )
        
        # Convert encoded batched examples into tensors
        preprocessed_dataset.set_format('torch')
        return preprocessed_dataset

class MMLU:
    '''
    Probably dont want to use it since its difficulty
    '''
    def __init__(self):
        data = load_dataset('cais/mmlu', "all")
        self.train_set = data['train']
        self.val_set = data['test']

        # Reserve 10% of the training set for few-shot prompting
        idx = np.random.choice(range(len(self.train_set)),  int(0.1 * len(self.train_set)))
        self.prompt_set = self.train_set[idx]
        train_idx = list(set(range(len(self.train_set))) - set(idx))
        self.train_set = self.train_set[train_idx]

    def get_eval_set(self, processor, template, batchsize=32):

        # Select only context and question columns
        preprocessed_dataset = self.val_set.map(
            lambda example: 
                {
                    'finputs': template(example['question']),
                }, 
                remove_columns=self.val_set.column_names
        )

        # Get max length
        max_len = 256

        # Tokenize the inputs
        def tokenize_function(examples):
            encoded = processor(
                    examples['finputs'],
                    padding='max_length',
                    max_length=max_len,
                )

            return {k: encoded[k] for k in encoded.keys() if k != "pixel_values"} 
                    
        preprocessed_dataset = preprocessed_dataset.map(
            tokenize_function, 
            remove_columns=[c for c in preprocessed_dataset.column_names if c != 'id'],
            batched = True,
            batch_size = batchsize
        )
        
        # Convert encoded batched examples into tensors
        preprocessed_dataset.set_format('torch')
        return preprocessed_dataset

class CommonsenseQA:

    def __init__(self):
        # Read raw dataset
        data = []
        with open("/scratch/t.tovi/datasets/train_rand_split.jsonl", 'r') as file:
            for line in file:
                data.append(json.loads(line))

        # Format question inputs
        fdata = []
        for entry in data:
            fchoices = ""
            for choice in entry['question']['choices']:
                fchoices += choice['label'] + " " + choice['text'] + '\n'
            
            fentry = {}
            fentry['id'] = entry['id']
            fentry['context'] = fchoices
            fentry['question'] = entry['question']['stem']
            fentry['answer'] = entry['answerKey']

            fdata.append(fentry)

        self.train_set = fdata

    def format_input_with_cot_prompt(self, eos_token, instruction, cot_prompts, cqa_template, cqa):

        # Prepend instruction
        formatted_input = instruction + '\n'

        # Randomly select a cot exmaple
        cot_example = cot_prompts[np.random.choice(len(cot_prompts))]
        cot_template = cqa_template(cot_example['Context'], cot_example['Question'], cot_example['Answer'])
        formatted_input += cot_template+'\n'

        # Append the gpt rationale
        formatted_input += "Rationale: " + cot_example['Rationale'] + eos_token

        # Apply template to the question
        inputs = cqa_template(cqa['context'], cqa['question'], cqa['answer'])
        formatted_input += instruction + '\n' + inputs + '\n'

        # Add Rationale prefix
        formatted_input += "Rationale:"

        return formatted_input

    def get_train_set(self, eos_token, instruction, template, cot_prompts, batchsize=32):
        '''
        Select only context and question columns
        '''
        preprocessed_dataset = []
        for batch in self.train_set:
            fbatch = {}
            fbatch['id'] = batch['id']

            fbatch['finputs'] = self.format_input_with_cot_prompt(eos_token, instruction, cot_prompts, template, batch)
            preprocessed_dataset.append(fbatch)

        return preprocessed_dataset

    def extract_answer(self, model_answers):
        '''
        Extract the succinct answers from the model generated texts
        '''
        pass

class CosmosQA:

    def __init__(self):
        # Read raw dataset
        data = []
        with open("/scratch/t.tovi/datasets/cosmosqa_train.csv", 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:
                id = line[0]
                context = line[1]
                question = line[2]
                choices = f"A {line[3]}\n B {line[4]}\n C {line[5]}\n D {line[6]}\n"
                mapping = {"0":"A", "1":"B", "2":"C", "3":"D"}
                answer = mapping[line[-1]]
                entry = {"id": id, "context": context, "question": question, "choices": choices, "answer": answer}
                data.append(entry)

        self.train_set = data

    def format_input_with_cot_prompt(self, eos_token, instruction, cot_prompts, cqa_template, cqa):

        # Prepend instruction
        formatted_input = instruction + '\n'

        # Randomly select a cot exmaple
        cot_example = cot_prompts[np.random.choice(len(cot_prompts))]
        cot_template = cqa_template(cot_example['Context'], cot_example['Question'], cot_example['Answer'])
        formatted_input += cot_template+'\n'

        # Append the gpt rationale
        formatted_input += "Rationale: " + cot_example['Rationale'] + eos_token

        # Format question prompt
        question_instruction = "Choose the most appropriate answer from below:\n"
        question = cqa["question"] + question_instruction + cqa['choices']

        # Apply template to the question
        inputs = cqa_template(cqa['context'], question, cqa['answer'])
        formatted_input += instruction + '\n' + inputs + '\n'

        # Add Rationale prefix
        formatted_input += "Rationale:"

        return formatted_input

    def get_train_set(self, eos_token, instruction, template, cot_prompts, batchsize=32):
        '''
        Select only context and question columns
        '''
        preprocessed_dataset = []
        for batch in self.train_set:
            fbatch = {}
            fbatch['id'] = batch['id']

            fbatch['finputs'] = self.format_input_with_cot_prompt(eos_token, instruction, cot_prompts, template, batch)
            preprocessed_dataset.append(fbatch)

        return preprocessed_dataset

    def extract_answer(self, model_answers):
        '''
        Extract the succinct answers from the model generated texts
        '''
        pass

class ARC:

    def __init__(self):
        # Read raw dataset
        data = []
        with open("/scratch/t.tovi/dataset/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl", 'r') as f:
            for line in f:
                data.append(json.loads(line))

        with open("/scratch/t.tovi/dataset/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl", 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        # Format choices
        self.train_set = []
        for entry in data:
            fentry = {}
            fentry['id'] = entry['id']
            fentry['question'] = entry['question']['stem']
            choices = [choice['label']+ " " + choice['text'] for choice in entry['question']['choices']]
            fentry['choices'] = "\n".join(choices)
            fentry['answer'] = entry['answerKey']
            self.train_set.append(fentry)


    def format_input_with_cot_prompt(self, eos_token, instruction, cot_prompts, qa_template, cqa):

        # Prepend instruction
        formatted_input = instruction + '\n'

        # Randomly select a cot exmaple
        cot_example = cot_prompts[np.random.choice(len(cot_prompts))]
        cot_template = qa_template(cot_example['Question'], cot_example['Answer'])
        formatted_input += cot_template+'\n'

        # Append the gpt rationale
        formatted_input += "Rationale: " + cot_example['Rationale'] + eos_token

        # Format question prompt
        question_instruction = "Choose the most appropriate answer from below:\n"
        question = cqa["question"] + question_instruction + cqa['choices']

        # Apply template to the question
        inputs = qa_template(question, cqa['answer'])
        formatted_input += instruction + '\n' + inputs + '\n'

        # Add Rationale prefix
        formatted_input += "Rationale:"

        return formatted_input

    def get_train_set(self, eos_token, instruction, template, cot_prompts, batchsize=32):
        '''
        Select only context and question columns
        '''
        preprocessed_dataset = []
        for batch in self.train_set:
            fbatch = {}
            fbatch['id'] = batch['id']

            fbatch['finputs'] = self.format_input_with_cot_prompt(eos_token, instruction, cot_prompts, template, batch)
            preprocessed_dataset.append(fbatch)

        return preprocessed_dataset

    def extract_answer(self, model_answers):
        '''
        Extract the succinct answers from the model generated texts
        '''
        pass

class VQA:

    def __init__(self):
        self.train_set = []

        # Read image captions
        captions = {}
        with open("/scratch/t.tovi/datasets/annotations/captions_train2017.json", "r") as f:
            data = json.load(f)

        # Group captions for the same image
        for anno in data['annotations']:
            if anno['image_id'] in captions:
                captions[anno['image_id']]['captions'].append(anno['caption'])
            else:
                captions[anno['image_id']] = {
                    'file_name': '',
                    'captions': [anno['caption']],
                }

        for image_info in data['images']:
            captions[image_info['id']]['file_name'] = image_info['file_name']

        # Read VQA answers
        answer_map = {}
        with open("/scratch/t.tovi/datasets/v2_mscoco_train2014_annotations.json", 'r') as f:
            answers = json.load(f)
            for answer in answers['annotations']:
                answer_map[answer['question_id']] = answer['answers']

        # Read VQA questions
        with open("/scratch/t.tovi/datasets/v2_OpenEnded_mscoco_train2014_questions.json", 'r') as f:
            questions = json.load(f)

        for question in questions['questions']:
            question['captions'] = captions[question['image_id']]
            question['answers'] = answer_map[question['question_id']]
            self.train_set.append(question)
        
    def format_input_with_cot_prompt(self, eos_token, instruction, cot_prompts, cqa_template, cqa):
        pass

    def get_train_set(self, eos_token, instruction, template, cot_prompts, batchsize=32):
        '''
        Select only context and question columns
        '''
        pass

    def extract_answer(self, model_answers):
        '''
        Extract the succinct answers from the model generated texts
        '''
        pass
