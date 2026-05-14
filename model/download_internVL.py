from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODEL_IDS = {
    "1b": "OpenGVLab/InternVL3_5-1B-HF",
    "2b": "OpenGVLab/InternVL3_5-2B-HF",
    "4b": "OpenGVLab/InternVL3_5-4B-HF",
    "8b": "OpenGVLab/InternVL3_5-8B-HF",
    "14b": "OpenGVLab/InternVL3_5-14B-HF",
    "38b": "OpenGVLab/InternVL3_5-38B-HF"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an InternVL3.5 model from Hugging Face.")
    parser.add_argument(
        "--size",
        choices=list(MODEL_IDS.keys()),
        default="8b",
        help="Model size to download. Default: 8b",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id = MODEL_IDS[args.size]
    print(f"Downloading: {model_id}")
    snapshot_download(
        repo_id=model_id,
        max_workers=16,
    )
    print("Done.")


if __name__ == "__main__":
    main()
