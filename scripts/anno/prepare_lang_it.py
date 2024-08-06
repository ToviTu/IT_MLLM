import json
import os
from llava.anno.evaluate_util import SQuAD, CommonsenseQA, StrategyQA, CosmosQA, ARC
import numpy as np

import random

def get_llava_train(factroy, anno_path):
    fac = factroy()
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    return fac.process_with_rationale(rationale=anno)

def sampling_prob(list_of_sizes):
    sizes = np.array(list_of_sizes, dtype=float)
    probs = 1 / sizes
    probs /= probs.sum()
    return probs

def sample_from_list(datasets, probs, size):
    assert len(datasets) == len(probs)
    counts = np.zeros(len(datasets), dtype=int)
    data = []
    while len(data) < size:
        idx = random.choices(range(len(datasets)), probs)
        counts[idx[0]] += 1
        data.append(random.choice(datasets[idx[0]]))
    return data, counts

def format_llava_train(data):
    outputs = []
    for d in data:
        split = d['finputs'].split("User: ")[-1]
        split_ = split.split("Assistant: ")

        input = split_[0]
        output = split_[1]

        outputs.append({
            "id": d['id'],
            "model": "Yi-1.5-34B",
            "conversations": [
                {
                    "from": "human",
                    "value": input
                },
                {
                    "from": "gpt",
                    "value": output
                }
            ]
        })
    
    return outputs

if __name__ == '__main__':

    size = 80000
    
    squad_train = get_llava_train(SQuAD, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_squad_rationale.json"))
    strategyqa_train = get_llava_train(StrategyQA, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_strategyqa_rationale.json"))
    commonsenseqa_train = get_llava_train(CommonsenseQA, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_commonsenseqa_rationale.json"))
    cosmosqa_train = get_llava_train(CosmosQA, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_cosmosqa_rationale.json"))
    arc_train = get_llava_train(ARC, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_ARC_rationale.json"))

    sizes = [len(squad_train), len(strategyqa_train), len(commonsenseqa_train), len(cosmosqa_train), len(arc_train)]
    probs = np.array([1/4, 3/16, 3/16, 3/16, 3/16])
    datasets = [squad_train, strategyqa_train, commonsenseqa_train, cosmosqa_train, arc_train]
    data, counts = sample_from_list(datasets, probs, size)
    print("Dataset distribution: ", counts)
    data = format_llava_train(data)

    with open(os.path.join(os.environ['STORAGE_DIR'], "datasets/llava/Yi_llava_train_balanced.json"), 'w') as f:
        json.dump(data, f)

