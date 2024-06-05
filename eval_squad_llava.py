from src.evaluate_util import SQuAD
from src.model_util import Llava
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompt_template import vicuna_cqa
import json
import tqdm
import torch

batch_size = 8

model = Llava()
model.eval()
data = SQuAD()
val_data = data.get_eval_set(vicuna_cqa)

#model.processor.padding_side = "left"

answers = []
for idx in tqdm.tqdm(range(0, len(val_data), batch_size)): 
    batch = val_data[idx: idx+batch_size]
    ids = batch['id']
    
    predicted_texts = model.generate(batch['finputs'], max_new_tokens=512)
    
    # Make sure to extract the new tokens only
    answers += [
        {'id': id, 'prediction_text': text.split('ASSISTANT:')[-1]} for id, text in zip(ids, predicted_texts)
    ]

    # Save periodically
    if len(answers) % (batch_size * 8) == 0:
        with open('/scratch/t.tovi/results/vicuna-1.5-7B_squad-prediction.json', 'w') as f:
            json.dump(answers, f)

# Save final predictions
with open(f'/scratch/t.tovi/results/vicuna-1.5-7B_squad-prediction.json', 'w') as f:
    json.dump(answers, f)