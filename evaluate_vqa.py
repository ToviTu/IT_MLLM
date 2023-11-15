from scripts.custom_evaluate import evaluate_vqa
from scripts.model import EvalModel
import os

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
    "checkpoint_path": f"{CHECKPOINT_DIR}/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt",
    "cross_attn_every_n_layers": 1,
    "precision": "bf16",
    "device": 0,
}

print(f"Loading Checkpoint from {CHECKPOINT_DIR}")
model = EvalModel(model_args)
print("Starting evaluation")
print(evaluate_vqa(model))
