import json
import os

from tqdm import tqdm
import argparse
import re
import transformers
import huggingface_hub
import tokenizers
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates, SeparatorStyle

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
import torch
import model_ARC_loader
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./data/eval/arc")
    parser.add_argument('--ckpt', type=str, default="model_v1")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--model-path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument('--summarize-strategy', type=str, default="pattern", choices=["llm", "pattern"])
    return parser.parse_args()



class EvalAIAnswerProcessor:
    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item.upper()  

class ARCAccuracyEvaluator:
    def __init__(self, args):
        self.model_path = args.model_path
        self.model_base = args.model_base
        self.answer_processor = EvalAIAnswerProcessor()
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, _, _ = load_pretrained_model(self.model_path, self.model_base, self.model_name)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_new_tokens = args.max_new_tokens

    def eval_pred_list(self, pred_list, gt_list):
        pred_scores = []
        all_answers = []
        regenerated_answers_list = []
        incorrect_answers = []

        for entry in tqdm(pred_list):
            pred_answer = self.answer_processor(entry["text"])
            gt_answer = gt_list[entry["question_id"]]
            if pred_answer != "":
                if pred_answer == gt_answer:
                    score = 1.0
                    result = "correct"
                else:
                    incorrect_answers.append(entry)
                    continue
            else:
                score = 0.0
                result = "incorrect"
                summarized_choice = "Empty"

            all_answers.append({
                "question_id": entry["question_id"],
                "predicted_answer": pred_answer,
                "ground_truth_answer": gt_answer,
                "result": result
            })
            pred_scores.append(score)

        # llm mode
        if args.summarize_strategy == "llm":
            # Regenerate incorrect answers and reevaluate
            regenerated_answers = self.regenerate_answers_llm(incorrect_answers)
            for entry in regenerated_answers:
                pred_answer = self.answer_processor(entry["text"])
                gt_answer = gt_list[entry["question_id"]]
                summarized_choice = pred_answer 

                if summarized_choice == gt_answer:
                    score = 1.0
                    result = "correct"
                else:
                    score = 0.0
                    result = "incorrect"

                regenerated_answers_list.append({
                    "question_id": entry["question_id"],
                    "pred_answer_text": entry["pred_answer_text"],
                    "regenerated_answer": pred_answer,
                    "ground_truth_answer": gt_answer,
                    "result": result
                })
                pred_scores.append(score)
        
        if args.summarize_strategy == "pattern":
            regenerated_answers = self.regenerate_answers_pattern(incorrect_answers)

            for entry in regenerated_answers:
                pred_answer = entry["text"]
                gt_answer = gt_list[entry["question_id"]]
                
                if pred_answer == gt_answer:
                    score = 1.0
                    result = "correct"
                else:
                    score = 0.0
                    result = "incorrect"

                regenerated_answers_list.append({
                    "question_id": entry["question_id"],
                    "original_text": entry["pred_answer_text"],  
                    "regenerated_answer": pred_answer, 
                    "ground_truth_answer": gt_answer,
                    "result": result
                })
                pred_scores.append(score)


        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy, all_answers, regenerated_answers_list
    
    def regenerate_answers_pattern(self, incorrect_answers):
        regenerated_answers = []
        pattern = re.compile(r'\b([ABCD])\.')

        for entry in incorrect_answers:
            text = entry["text"]
            match = pattern.search(text)
            
            if match:
                pred_answer_text = match.group(1)
            else:
                pred_answer_text = "None" 

            regenerated_answers.append({
                "question_id": entry["question_id"],
                "text": pred_answer_text,
                "pred_answer_text": text
            })

        return regenerated_answers
        

    def regenerate_answers_llm(self, incorrect_answers):
        regenerated_answers = []
        for entry in tqdm(incorrect_answers, total=len(incorrect_answers)):
            pred_answer_text = entry["text"]
            prompt = (
                f"Given the answer '{pred_answer_text}', select the best choice: A, B, C, or D. Only respond with the correct letter."
            )
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=0.1,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    max_new_tokens=1,
                    use_cache=True
                )
                
            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            regenerated_answers.append({
                "question_id": entry["question_id"],
                "prompt": prompt,
                "text": output,
                "pred_answer_text": pred_answer_text
            })
            

        return regenerated_answers

if __name__ == '__main__':
    args = parse_args()

    src = '/scratch/peterni/wustl/IT_MLLM/llava/eval/arc_answers.jsonl'
    test_split = '/scratch/peterni/wustl/IT_MLLM/datasets/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl'
    dst = os.path.join(args.dir, 'answers_upload', args.split, f'{args.ckpt}_eval.json')
    regenerated_dst = os.path.join(args.dir, 'answers_upload', args.split, f'{args.ckpt}_regenerated.json')

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Load predictions
    results = []
    for line in open(src):
        results.append(json.loads(line))

    # Load ground truth
    gt_answers = {}
    for line in open(test_split):
        entry = json.loads(line)
        gt_answers[entry['id']] = entry['answerKey']

    print(f'total results: {len(results)}, total questions: {len(gt_answers)}')

    # Evaluate
    evaluator = ARCAccuracyEvaluator(args)
    accuracy, all_answers, regenerated_answers_list = evaluator.eval_pred_list(results, gt_answers)

    # Save evaluation results
    eval_results = {
        'accuracy': accuracy,
        'total_questions': len(gt_answers),
        'total_predictions': len(results),
        'all_answers': all_answers
    }

    with open(dst, 'w') as f:
        json.dump(eval_results, f)

    print(f'Evaluation results saved to {dst}')
    
    regnerated_results = {
        'total_questions': len(regenerated_answers_list),
        'regenerated_answers': regenerated_answers_list
    }
    

    with open(regenerated_dst, 'w') as f:
            json.dump(regnerated_results, f)

    print(f'Regenerated answers saved to {regenerated_dst}')