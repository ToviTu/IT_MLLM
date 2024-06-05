from src.evaluate_util import SQuAD
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompt_template import vicuna_cqa
import json
import tqdm
import torch

batch_size = 8

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5", 
    device_map='auto',
    cache_dir="/scratch/t.tovi/models/"
)
model.eval()
data = SQuAD()
val_data = data.get_eval_set(vicuna_cqa)

tokenizer.padding_side = "left"

answers = []
for idx in tqdm.tqdm(range(0, len(val_data), batch_size)): 
    batch = val_data[idx: idx+batch_size]
    ids = batch['id']

    encoded_text = tokenizer(text=batch['finputs'], padding=True, return_tensors='pt').to(0)

    preds = model.generate(**encoded_text, max_new_tokens=512)
    predicted_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
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