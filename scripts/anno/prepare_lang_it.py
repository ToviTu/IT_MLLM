import json
import os
from llava.anno.evaluate_util import SQuAD, CommonsenseQA, StrategyQA, CosmosQA, ARC

import random

def get_llava_train(factroy, anno_path):
    fac = factroy()
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    return fac.process_with_rationale(rationale=anno)
    

if __name__ == '__main__':
    
    squad_train = get_llava_train(SQuAD, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_squad_rationale.json"))
    strategyqa_train = get_llava_train(StrategyQA, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_strategyqa_rationale.json"))
    commonsenseqa_train = get_llava_train(CommonsenseQA, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_commonsenseqa_rationale.json"))
    cosmosqa_train = get_llava_train(CosmosQA, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_cosmosqa_rationale.json"))
    arc_train = get_llava_train(ARC, os.path.join(os.environ['STORAGE_DIR'], "results/anno/Yi_ARC_rationale.json"))

    print("SQuAD: ", len(squad_train))
    print(random.choice(squad_train))

    print("StrategyQA: ", len(strategyqa_train))
    print("StrategyQA: ", random.choice(strategyqa_train))

    print("CommonsenseQA: ", len(commonsenseqa_train))
    print("CommonsenseQA: ", random.choice(commonsenseqa_train))

    print("CosmosQA: ", len(cosmosqa_train))
    print("CosmosQA: ", random.choice(cosmosqa_train))

    print("ARC: ", len(arc_train))
    print("ARC: ", random.choice(arc_train))

    # Todo random sampling training data from each dataset with weights