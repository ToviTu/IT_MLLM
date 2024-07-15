import json
import argparse

parser = argparse.ArgumentParser(description='Process JSONL file and output JSON.')
parser.add_argument('--input_file', type=str, default='/storage1/chenguangwang/Active/vision_share/dataset/eval/vizwiz/llava_test.jsonl')
parser.add_argument('--output_file', type=str, default='/storage1/chenguangwang/Active/vision_share/dataset/eval/vizwiz/llava_test_new.jsonl')
parser.add_argument('--prompt', type=str, default='\nASSISTANT:')
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file


def question_reformat(text):
    new_text = text.split('\n')[0] + "\nAnswer the question using a single word or phrase." + args.prompt
    return new_text

data = []
# Read each line from the JSONL file and parse as JSON
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()  # Remove leading/trailing whitespace and newline
        if line:
            json_obj = json.loads(line)
            data.append(json_obj)

converted_data = []

for item in data:
    question_id = item['question_id']
    image = item['image']
    text = question_reformat(item['text'])
    category = item['category']
    new_item = {'question_id': question_id, 'image': image, 'text': text, 'category': category}
    converted_data.append(new_item)

with open(output_file, 'w') as fout:
    for item in converted_data:
        json.dump(item, fout)
        fout.write('\n')