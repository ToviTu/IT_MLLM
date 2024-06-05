from src.evaluate_util import CosmosQA
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
data = CosmosQA()
with open("/home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/GPT_cosmosQA_rationale.json", 'r') as f:
    cot_examples = json.load(f)
instruction = """You are a helpful agent in answering questions. You will be provided with a context, a question, and its answer,  and your job is to supply a detailed step-by-step explanation. You need to first paraphrase the problem, state the relevant premise according to the context, deduct facts one at a step, and finally give the reason why the answer is correct. If you disagree, please reply with "I disagree". """

train_data = data.get_train_set(model.processor.eos_token, instruction, llm_cqa, cot_examples, batchsize=batch_size)

answers = []
for idx in tqdm.tqdm(range(0, len(train_data), batch_size)): 
    batch = train_data[idx: idx+batch_size]
    finputs = [each['finputs'] for each in batch]
    ids = [each['id'] for each in batch]

    predicted_texts = model.generate(finputs, skip_special_tokens=False, max_new_tokens=512, do_sample=True, top_p=0.7, temperature=0.15)

    #Keep the new tokens
    predicted_texts = [text.split(model.processor.eos_token)[1] for text in predicted_texts]

    # Make sure to extract the new tokens only
    answers += [
        {'id': id, 'prediction_text': text} for id, text in zip(ids, predicted_texts)
    ]

    # Save periodically
    if len(answers) % (batch_size * 8) == 0:
        with open('/scratch/t.tovi/results/Yi_cosmosqa_rationale.json', 'w') as f:
            json.dump(answers, f)

# Save final predictions
with open('/scratch/t.tovi/results/Yi_cosmosqa_rationale.json', 'w') as f:
    json.dump(answers, f)
