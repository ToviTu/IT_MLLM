from scripts.model import EvalModel
import os
from scripts.datasets import VQA_dataset
from scripts.datasets import SQUAD_dataset
from huggingface_hub import hf_hub_download

HF_TOKEN = "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"
CHECKPOINT_DIR = (
    "/mnt/d/models/"
    if os.environ.get("CHECKPOINT_DIR") == None
    else os.environ["CHECKPOINT_DIR"]
)

model_args = {
    "vision_encoder_path": "ViT-L-14",
    "vision_encoder_pretrained": "openai",
    "lm_path": "anas-awadalla/mpt-1b-redpajama-200b",
    "lm_tokenizer_path": "anas-awadalla/mpt-1b-redpajama-200b",
    "checkpoint_path": f"{CHECKPOINT_DIR}/OpenFlamingo-3B-vitl-mpt1b-ft-squad/checkpoint.pt",
    "cross_attn_every_n_layers": 1,
    "precision": "bf16",
    "device": 0,
}


checkpoint_path = hf_hub_download(
    "ToviTu/fine-tuned-nl-flamingo",
    "checkpoint.pt",
    local_dir=CHECKPOINT_DIR + "fine-tuned-nl-flamingo",
    cache_dir=CHECKPOINT_DIR + "fine-tuned-nl-flamingo",
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)

print(f"Loading Checkpoint from {CHECKPOINT_DIR}")
model = EvalModel(model_args)

data = SQUAD_dataset()
data.infer(
    model.model, model.image_processor, model.tokenizer, "/external/", early_stop=1000
)

import json
from evaluate import load

with open("/external/squad_resutls.json", "r") as f:
    predictions = json.load(f)
with open("/external/squad_references.json", "r") as f:
    references = json.load(f)

squad_metric = load("squad")
results = squad_metric.compute(predictions=predictions, references=references)
print(results)
