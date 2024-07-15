import json
import re
import argparse

parser = argparse.ArgumentParser(description='Process JSONL file and output JSON.')
parser.add_argument('--input_file', type=str, default='/storage1/chenguangwang/Active/vision_share/dataset/eval/vizwiz/answers/vicuna-7b-projector-vicuna_v1-conv.jsonl')
parser.add_argument('--output_file', type=str, default='answers_upload/vizwiz_predictions_vicuna-7b-projector-vicuna_v1-conv.json')
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

data = []
# Read each line from the JSONL file and parse as JSON
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()  # Remove leading/trailing whitespace and newline
        if line:
            json_obj = json.loads(line)
            data.append(json_obj)

converted_data = {}

def parse_prompt(prompt):
    # Split the prompt based on '\n' to get the part after '\n'
    parts = re.split('[\r\n]+', prompt) 
    
    if len(parts) > 1: 
        direct_answer = parts[1].strip()
    else:
        direct_answer = " "

    # Find the letter grade answer from the choices
    letter_grade = None
    answer = direct_answer.split('answer is: ')
    if len(answer) > 1: 
        letter_grade = answer[1][0] 
        if(letter_grade == 'A'): letter_grade=0
        elif(letter_grade == 'B'): letter_grade=1
        elif(letter_grade == 'C'): letter_grade=2
        elif(letter_grade == 'D'): letter_grade=3
    return letter_grade, direct_answer

# Iterate through each item in the JSON data
for item in data:
    question_id = item['question_id']
    prompt = item['text']
    
    # Parse the prompt to get multiple choice prediction and direct answer
    letter_grade, direct_answer = parse_prompt(prompt)
    
    # Store the predictions in the desired format
    converted_data[question_id] = {
        'multiple_choice': letter_grade,  # Assuming letter grade is the multiple choice prediction
        #'direct_answer': direct_answer
    }

# Write the converted data to a new JSON file

with open(output_file, 'w') as f:
    json.dump(converted_data, f, indent=4)

print(f'Converted data saved to {output_file}')