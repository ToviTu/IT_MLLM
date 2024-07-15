import json
import argparse

# Function to parse the prompt and extract answer
def parse_prompt(prompt):
    final_answer = ""
    answer = prompt.split('answer is: ')
    if len(answer) > 1:
        final_answer = answer[1].replace('.', '')
    return final_answer


# Main function to process the JSONL file
def process_jsonl(input_file, output_file, num_lines=200):
    converted_data = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            item = json.loads(line)

            question_id = item['question_id']
            image_id = filename = f"VizWiz_test_{question_id:08d}.jpg"
            raw_answer = item['text']

            # Parse the prompt to get the final answer
            final_answer = parse_prompt(raw_answer)

            # Store the parsed data in the desired format
            answer_pair = {'image': image_id, 'answer': final_answer}
            converted_data.append(answer_pair)

    # Write the converted data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(converted_data, f)

    print(f'Converted data saved to {output_file}')

# Command line argument parser setup
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process JSONL file and output JSON.')
    parser.add_argument('--input_file', type=str, default='/storage1/chenguangwang/Active/vision_share/dataset/eval/vizwiz/answers/vicuna-7b-projector-vicuna_v1-conv.jsonl')
    parser.add_argument('--output_file', type=str, default='answers_upload/vizwiz_predictions_vicuna-7b-projector-vicuna_v1-conv.json')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    input_file = args.input_file
    output_file = args.output_file

    # Process the JSONL file
    process_jsonl(input_file, output_file, num_lines=200)