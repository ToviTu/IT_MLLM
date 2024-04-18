from src.evaluate_util import SQuAD
from src.model_util import Yi
from src.prompt_template import llm_cqa
import json
import tqdm
import torch

# Parameters
batch_size = 4

# Model
model = Yi()
model.eval()
model.processor.padding_side = 'left'

# Data
data = SQuAD()
with open("/home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/GPT_squad_rationale.json", 'r') as f:
    cot_examples = json.load(f)
instruction = '''You are a helpful agent who assists the user in answering a wide range of problems. You will be given a question and an answer, and your job is to provide the best explanation for the given answer by following the given example. You should first identify the problem and give step-by-step reasoning in detail. Begin your respond with "Rationale:". If you disagree with the answer, please respond with "I disagree'''

train_data = data.get_train_set(instruction, model.processor, llm_cqa, cot_examples, batchsize=batch_size)


answers = []
for idx in tqdm.tqdm(range(0, len(train_data), batch_size)): 
    batch = train_data[idx: idx+batch_size]

    predicted_texts = model.generate(batch['finputs'])

    # Make sure to extract the new tokens only
    answers += [
        {'id': id, 'prediction_text': text.split('Rationale:')[-1]} for id, text in zip(batch['id'], predicted_texts)
    ]

    # Save periodically
    if len(answers) % (batch_size * 8) == 0:
        with open('/scratch/t.tovi/results/Yi_squad_rationale.json', 'w') as f:
            json.dump(answers, f)

# Save final predictions
with open('/scratch/t.tovi/results/Yi_squad_rationale.json', 'w') as f:
    json.dump(answers, f)
