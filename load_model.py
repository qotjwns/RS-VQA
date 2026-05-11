import os
from huggingface_hub import snapshot_download


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
MODEL_ID = "OpenGVLab/InternVL3_5-8B"
SAVE_DIR = "/workspace/models/InternVL3_5-8B"

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=SAVE_DIR,
    resume_download=True,
    max_workers=16,
)

print(f"Downloaded to: {SAVE_DIR}")