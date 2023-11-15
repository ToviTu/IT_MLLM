from huggingface_hub import hf_hub_download
import os

HF_TOKEN = "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"
CHECKPOINT_DIR = (
    "/mnt/d/models/"
    if os.environ.get("CHECKPOINT_DIR") == None
    else os.environ["CHECKPOINT_DIR"]
)

checkpoint_path = hf_hub_download(
    "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    "checkpoint.pt",
    local_dir=CHECKPOINT_DIR + "OpenFlamingo-3B-vitl-mpt1b",
    cache_dir=CHECKPOINT_DIR + "OpenFlamingo-3B-vitl-mpt1b",
    local_dir_use_symlinks=False,
    token=HF_TOKEN,
)
print(checkpoint_path)
## openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt
