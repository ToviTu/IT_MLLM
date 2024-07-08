import json
import os

storage_dir = os.environ.get('STORAGE_DIR', '/default/storage/path')
working_dir = os.environ.get('WORKING_DIR', '/default/working/path')

input_path = os.path.join(working_dir, 'llava/eval/data/inference/vicuna/strategyqa_answers.jsonl')
output_path = os.path.join(working_dir, 'llava/eval/data/inference/vicuna/strategyqa_answers_reformat.json')

transformed_data = {}

with open(input_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        question_id = data['question_id']
        text = data['text'].strip('.')

        if text.lower() == 'true':
            answer = True
        elif text.lower() == 'false':
            answer = False
        else:
            continue 

        transformed_data[question_id] = {
            'answer': answer,
            'decomposition': [], 
            'paragraphs': []  
        }

with open(output_path, 'w') as outfile:
    json.dump(transformed_data, outfile, indent=4)

print(f"Data transformed and saved to {output_path}")