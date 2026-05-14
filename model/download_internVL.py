import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

#모델 파라미터수 선택
MODEL_ID = "OpenGVLab/InternVL3_5-14B-HF"

snapshot_download(
    repo_id=MODEL_ID,
    max_workers=16,
)
