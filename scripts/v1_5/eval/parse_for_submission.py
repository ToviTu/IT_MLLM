import json
import argparse


# Function to parse the prompt and extract answer
def parse_prompt(prompt):
    final_answer = ""
    answer = prompt.split('answer is: ')
    if len(answer) > 1:
        final_answer = answer[1]
    return final_answer

# Main function to process the JSONL file
def process_jsonl(input_file, output_file, num_lines=200):
    converted_data = []

    with open(input_file, 'r') as f:
        raw_data = json.load(f)
    print(len(raw_data))
    data=raw_data[:200] 
    print(len(data))
    
    for item in data: 
        question_id = item['question_id']
        raw_answer = item['answer']

        # Parse the prompt to get the final answer
        final_answer = parse_prompt(raw_answer)

        # Store the parsed data in the desired format
        question_data = {'question_id': question_id, 'answer': final_answer}
        converted_data.append(question_data)

    # Write the converted data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(converted_data, f)

    print(f'Converted data saved to {output_file}')

# Command line argument parser setup
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process JSONL file and output JSON.')
    parser.add_argument('--input_file', type=str, default='/storage1/chenguangwang/Active/vision_share/dataset/eval/vqav2/answers_upload/test_200_lines/llava-vicuna-7b-lit-plain-conv.json')
    parser.add_argument('--output_file', type=str, default='vqav2_predictions_test_llava-vicuna-7b-lit-plain-conv.json')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    input_file = args.input_file
    output_file = args.output_file

    # Process the JSONL file
    process_jsonl(input_file, output_file, num_lines=200)
