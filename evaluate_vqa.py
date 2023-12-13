from scripts.custom_evaluate import evaluate_vqa
from scripts.model import EvalModel
from huggingface_hub import hf_hub_download
import os

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
print("Starting evaluation")
print(evaluate_vqa(model, num_samples=os.environ["EARLY_STOP"]))
