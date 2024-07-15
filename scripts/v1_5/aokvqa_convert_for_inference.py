import json
import argparse

parser = argparse.ArgumentParser(description='Process JSONL file and output JSON.')
parser.add_argument('--input_file', type=str, default='aokvqa_v1p0_val.json')
parser.add_argument('--output_file', type=str, default='aokvqa_val_vicuna_prompt.jsonl')
parser.add_argument('--prompt', type=str, default='\nASSISTANT:')
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

# Function to add prompt to question field
def add_prompt_and_choices(obj):
    prompt = ' Answer with the option’s letter from the given choices directly.'
    choices_with_letters = [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(obj['choices'])]
    choices_text = ', '.join(choices_with_letters)
    obj['text'] += prompt + ' Choices: ' + choices_text + args.prompt
    return obj

# Function to rename key within JSON object
def rename_key_in_json(json_obj, old_key, new_key):
    for obj in json_obj:
        if old_key in obj:
            obj[new_key] = obj.pop(old_key)
    return json_obj

# Function to cast "image" value from int to str
def cast_image_to_str(obj):
    if 'image' in obj:
        obj['image'] = str(obj['image'])
    return obj

def format_image(image_number):
    return f"{image_number:012}.jpg"

# Read the JSON file
with open(input_file, 'r') as json_file:
    data = json.load(json_file)

# Modify each object
modified_data = rename_key_in_json(data, "image_id", "image")
modified_data = rename_key_in_json(data, "question", "text")
modified_data = [add_prompt_and_choices(obj) for obj in modified_data]
modified_data = [cast_image_to_str(obj) for obj in modified_data]


for obj in modified_data:
    if 'image' in obj:
        obj['image'] = format_image(int(obj['image']))  # Convert to integer if necessary

# Write to JSONL file
with open(output_file, 'w') as jsonl_file:
    for item in modified_data:
        jsonl_file.write(json.dumps(item) + '\n')

