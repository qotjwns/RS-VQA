from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODEL_IDS = {
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2b": "Qwen/Qwen3.5-2B",
    "4b": "Qwen/Qwen3.5-4B",
    "9b": "Qwen/Qwen3.5-9B",
    "27b": "Qwen/Qwen3.5-27B",
    "35b": "Qwen/Qwen3.5-35B-A3B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Qwen3.5 model from Hugging Face.")
    parser.add_argument(
        "--size",
        choices=list(MODEL_IDS.keys()),
        default="9b",
        help="Model size to download. Default: 9b",
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
