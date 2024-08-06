import json
import os
import random
import shutil
import numpy as np

from llava.anno.evaluate_util import VQAv2, AOKvqa


def get_llava_train(factroy, anno_path, output_dir):
    fac = factroy()
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    return fac.process_with_rationale(rationale=anno, output_dir=output_dir)

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
            "image": d['image'],
            "model": "LLaVA-1.5-34B",
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
    
    vqa_train = get_llava_train(
        VQAv2, 
        os.path.join(os.environ['STORAGE_DIR'], "results/anno/Llava_vqav2_rationale.json"),
        os.path.join(os.environ['STORAGE_DIR'], "results/anno/images")
    )
    okvqa_train = get_llava_train(
        AOKvqa, 
        os.path.join(os.environ['STORAGE_DIR'], "results/anno/Llava_aokvqa_rationale.json"),
        os.path.join(os.environ['STORAGE_DIR'], "results/anno/images")
    )
    with open(os.path.join(os.environ['STORAGE_DIR'], "results/anno/complex_reasoning_77k.json"), 'r') as f:
        llava_train = json.load(f)

    for instance in llava_train:
        shutil.copyfile(
            os.path.join(os.environ['STORAGE_DIR'], "datasets/train2017", instance['image']),
            os.path.join(os.environ['STORAGE_DIR'], "results/anno/images", instance['image'])
        )

    vqav2_data = format_llava_train(vqa_train)
    aokvqa_data = format_llava_train(okvqa_train)
    llava_data = random.sample(llava_train, size-len(vqav2_data)-len(aokvqa_data))    
    print(len(vqav2_data), len(aokvqa_data), len(llava_data))
    data = vqav2_data + aokvqa_data + llava_data

    with open(os.path.join(os.environ['STORAGE_DIR'], "datasets/llava/Llava_llava_train_balanced.json"), 'w') as f:
        json.dump(data, f)