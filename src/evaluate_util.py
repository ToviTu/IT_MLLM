from datasets import load_dataset
from datasets import load_metric
import numpy as np

class SQuAD:

    def __init__(self):
        data = load_dataset("squad")
        self.train_set = data['train']
        self.val_set = data['validation']
        self.metric = load_metric("squad")
    
    def format_input_with_cot_prompt(self, instruction, cot_prompts, cqa_template, cqa):

        # Prepend instruction
        formatted_input = instruction + '\n'

        # Randomly select a cot exmaple
        cot_example = cot_prompts[np.random.choice(len(cot_prompts))]
        cot_template = cqa_template(cot_example['Context'], cot_example['Question'], cot_example['Answer'])
        formatted_input += cot_template+'\n'

        # Append the gpt rationale
        formatted_input += "Rationale: " + cot_example['Rationale'] +'\n'

        # Apply template to the question
        inputs = cqa_template(cqa['context'], cqa['question'], cqa['answers']['text'][0])
        formatted_input += inputs+'\n'

        # Add Rationale prompt
        formatted_input += "Rationale:"

        return formatted_input
    
    def get_train_set(self, instruction, processor, template, cot_prompts, batchsize=32):
        '''
        Select only context and question columns
        '''
        preprocessed_dataset = self.train_set.map(
            lambda example: 
                {
                    'finputs': self.format_input_with_cot_prompt(instruction, cot_prompts, template, example),
                }, 
                remove_columns=[c for c in self.train_set.column_names if c != 'id']
        )
        
        # Convert encoded batched examples into tensors
        preprocessed_dataset.set_format('torch')
        return preprocessed_dataset

    def get_eval_set(self, processor, template, batchsize=32):

        # Select only context and question columns
        preprocessed_dataset = self.val_set.map(
            lambda example: 
                {
                    'finputs': template(example['question'], example['context']),
                    #'len': len(template(example['question'], example['context']).split())
                }, 
                remove_columns=[c for c in self.val_set.column_names if c != 'id']
        )

        # Get max length
        max_len = 1024

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

