from src.evaluate_util import GSM8K
from src.model_util import *
from src.model_util import *
from src.prompt_template import llava_qa
import json
import tqdm
import numpy as np
batch_size = 16

model = Llava(quantization='bfloat16')
data = GSM8K()
val_data = data.get_eval_set(model.processor, llava_qa)

answers = []
for idx in tqdm.tqdm(range(0, len(val_data), batch_size)): 
    batch = val_data[idx: idx+batch_size]
    ids = batch['input_ids']
    encoded_text = {
        'input_ids': batch['input_ids'].to(model.device),
        'attention_mask': batch['attention_mask'].to(model.device)
    }

    with torch.no_grad():
        preds = model.model.generate(**encoded_text, pixel_values=None, max_new_tokens=1024)
        predicted_texts = model.processor.batch_decode(preds, skip_special_tokens=True)

    answers += [
        {'id': id.cpu().numpy().tolist(), 'prediction_text': text} for id, text in zip(ids, predicted_texts)
    ]

with open('/scratch/t.tovi/results/GSM8k_LLaVA_prediction.json', 'w') as f:
    json.dump(answers, f)
