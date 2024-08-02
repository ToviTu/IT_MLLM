"""
This files selects  the output logs from lmms-eval framework to
"""

import json
import argparse

parser = argparse.ArgumentParser(description='Process JSONL file and output JSON.')
parser.add_argument('--input_file', type=str, default='/scratch/yu.zihao/llava-vicuna-7b-vit_vqav2/0730_1719_llava...b-vit_llava_model_args_b919f2/vqav2_val_finetuned.json')
parser.add_argument('--output_file', type=str, default='/scratch/yu.zihao/llava-vicuna-7b-vit_vqav2/0730_1719_llava...b-vit_llava_model_args_b919f2/vqav2_val_finetuned_submission.json')
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

with open(input_file, "r") as f:
    data = json.load(f)
    logs = data['logs']

file_for_submission = []
for log in logs:
    question_id = log['doc']['question_id']
    answer = log['filtered_resps'][0]
    file_for_submission.append({"question_id": question_id, "answer": answer})

with open(output_file, 'w') as f:
    json.dump(file_for_submission, f)
