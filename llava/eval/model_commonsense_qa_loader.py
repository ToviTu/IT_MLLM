import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import pdb

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import math
storage_dir = os.environ.get('STORAGE_DIR', '/default/storage/path')
working_dir = os.environ.get('WORKING_DIR', '/default/working/path')
question_file_path = os.path.join(storage_dir, "IT_MLLM/datasets/commonsense_qa/test-00000-of-00001.parquet")
answer_file_path = os.path.join(storage_dir, "IT_MLLM/llava/eval/commonsense_qa_answer.jsonl")

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n) 
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class for CommonsenseQA
class CommonsenseQADataset(Dataset):
    def __init__(self, dataframe, tokenizer, model_config):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.model_config = model_config

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        question = row["question"]
        choices = row["choices"]
        labels = choices["label"]
        options = choices["text"]
        options = list(options)  
        labels = list(labels)

        options_with_labels = [f"{label}. {option}" for label, option in zip(labels, options)]
        
        prompt = f"Question: {question}\nOptions: {', '.join(options_with_labels)}\n Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print("prompt_final: ", prompt)

        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids

        return input_ids, prompt, row["id"]

    def __len__(self):
        return len(self.dataframe)


def collate_fn(batch):
    input_ids, options, ids = zip(*batch)
    input_ids = torch.cat(input_ids, dim=0)
    return input_ids, options, ids


# DataLoader
def create_data_loader(dataframe, tokenizer, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CommonsenseQADataset(dataframe, tokenizer, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    dataframe = pd.read_parquet(os.path.expanduser(args.question_file))
    dataframe = get_chunk(dataframe, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)
        
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(dataframe, tokenizer, model.config)

    for input_ids, prompt, ids in tqdm(data_loader, total=len(dataframe)):
        qid = ids[0]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": qid,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=question_file_path)
    parser.add_argument("--answers-file", type=str, default=answer_file_path)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
