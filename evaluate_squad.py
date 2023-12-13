from genericpath import exists
from scripts.model import EvalModel
import os
from scripts.datasets import SQUAD_dataset
from huggingface_hub import hf_hub_download
from scripts.download_dataset import DATASET_DIR

HF_TOKEN = "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"
CHECKPOINT_DIR = (
    "/mnt/d/models/"
    if os.environ.get("CHECKPOINT_DIR") == None
    else os.environ["CHECKPOINT_DIR"]
)

model_args = {
    "vision_encoder_path": "ViT-L-14",
    "vision_encoder_pretrained": "openai",
    "lm_path": os.environ["LM_PATH"],
    "lm_tokenizer_path": os.environ["LM_PATH"],
    "checkpoint_path": f"{CHECKPOINT_DIR}{os.environ['MODEL_PT']}/{os.environ['TYPE']}",
    "cross_attn_every_n_layers": 1,
    "precision": "bf16",
    "device": 0,
}

if not os.path.exists(f"{CHECKPOINT_DIR}{os.environ['MODEL_PT']}/{os.environ['TYPE']}"):
    checkpoint_path = hf_hub_download(
        f"ToviTu/{os.environ['MODEL_PT']}",
        os.environ["TYPE"],
        local_dir=CHECKPOINT_DIR + os.environ["MODEL_PT"],
        cache_dir=CHECKPOINT_DIR + os.environ["MODEL_PT"],
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )

print(
    f"Loading Checkpoint from {CHECKPOINT_DIR}{os.environ['MODEL_PT']}/{os.environ['TYPE']}"
)
model = EvalModel(model_args)

data = SQUAD_dataset()
data.infer(
    model.model,
    model.image_processor,
    model.tokenizer,
    DATASET_DIR,
    early_stop=int(os.environ["EARLY_STOP"]),
)

import json
from evaluate import load

with open(f"{DATASET_DIR}squad_resutls.json", "r") as f:
    predictions = json.load(f)
with open(f"{DATASET_DIR}squad_references.json", "r") as f:
    references = json.load(f)

squad_metric = load("squad")
results = squad_metric.compute(predictions=predictions, references=references)
print(results)
