import json

DATA_DIR = "/scratch/t.tovi/results/"
DEFAULT_PATHS = [
    "Yi_ARC_rationale.json",
    "Yi_commonsenseqa_rationale.json",
    "Yi_cosmosqa_rationale.json",
    "Yi_squad_rationale.json",
    "Yi_strategyqa_rationale_f.json",
]

TEMPLATE = {
    "id": None,
    "model": None,
    "conversations": [
        {
            "from": "human",
            "value": None,
        },
        {
            "from": "gpt",
            "value": None
        }
    ],

}

def format_data(list_of_paths: str, output_dir: str) -> None:
    all_data = []
    for path in list_of_paths:
        with open(path, 'r') as f:
            


