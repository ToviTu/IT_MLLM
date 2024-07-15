import json

input_file = '/storage1/chenguangwang/Active/vision_share/dataset/eval/vizwiz/llava_test.jsonl'
output_file = '/storage1/chenguangwang/Active/vision_share/dataset/eval/vizwiz/llava_test_new_vicuna.jsonl'


def question_reformat(text):
    #new_text = text.split('\n')[0] + "\nAnswer the question using a single word or phrase." + "\nASSISTANT:"
    #new_text = text.split('\n')[0] + "\nAnswer the question using a single word or phrase." + "\nAssistant: The answer is /"
    new_text = text + "\nASSISTANT:"
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