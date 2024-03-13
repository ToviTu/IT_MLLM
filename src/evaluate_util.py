from datasets import load_dataset
import numpy as np

class SQuAD:

    def __init__(self):
        data = load_dataset("squad")
        self.train_set = data['train']
        self.val_set = data['validation']

        # Reserve 10% of the training set for few-shot prompting
        idx = np.random.choice(range(len(self.train_set)), int(0.1 * len(self.train_set)))
        self.prompt_set = self.train_set[idx]
        train_idx = list(set(range(len(self.train_set))) - set(idx))
        self.train_set = self.train_set[train_idx]

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

