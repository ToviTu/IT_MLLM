from datasets import load_dataset

class SQuAD:

    def __init__(self):
        data = load_dataset("squad")
        self.train_set = data['train']
        self.val_set = data['validation']

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
        
        # Conver encoded batched examples into tensors
        preprocessed_dataset.set_format('torch')
        return preprocessed_dataset

