from src.evaluate_util import SQuAD
from src.model_util import *
from src.prompt_template import llava_cqa
import json
import tqdm

batch_size = 16

model = Llama2()
data = SQuAD()
val_data = data.get_eval_set(model.processor, llava_cqa)

answers = []
for idx in tqdm.tqdm(range(0, len(val_data), batch_size)): 
    batch = val_data[idx: idx+batch_size]
    ids = batch['id']
    encoded_text = {
        'input_ids': batch['input_ids'].to(model.device),
        'attention_mask': batch['attention_mask'].to(model.device)
    }

    with torch.no_grad():
        preds = model.model.generate(**encoded_text, max_new_tokens=1024)
        predicted_texts = model.processor.batch_decode(preds, skip_special_tokens=True)
    
    # Make sure to extract the new tokens only
    answers += [
        {'id': id, 'prediction_text': text.split('Assistant:')[-1]} for id, text in zip(ids, predicted_texts)
    ]

    # Save periodically
    if len(answers) % (batch_size * 8) == 0:
        with open('/scratch/t.tovi/results/squad_prediction.json', 'w') as f:
            json.dump(answers, f)

# Save final predictions
with open(f'/scratch/t.tovi/results/llama2-7B_squad-prediction.json', 'w') as f:
    json.dump(answers, f)
