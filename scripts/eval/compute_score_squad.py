import json
import argparse
from src.evaluate_util import SQuAD

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process the file path.')
    
    # Add the 'file' argument
    parser.add_argument('--file', type=str, help='The file path to process')
    parser.add_argument('--sep', type=str, help='The file path to process')
    
    # Parse the arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    with open(args.file, 'r') as f:
        preds = json.load(f)
        data = SQuAD()
        # Extract answers
        preds = [{'id': each['id'], 'prediction_text': each['prediction_text'].split(args.sep)[-1]} for each in preds]
        
        # Compute the score
        p, gd, score = data.extract_answer(preds)
        print(score)