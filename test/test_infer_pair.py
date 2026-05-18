from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    logging as transformers_logging,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# Test-only defaults. All paths are resolved from the repository root.
MODEL_ID = "OpenGVLab/InternVL3_5-38B-HF"
SAMPLE_INDEX = 300
MAX_NEW_TOKENS = 128
ANNOTATION_PATH = Path(
    "data/coding/muti_task_data/test_task_data/count_build.json"
)
LOCAL_DATA_ROOT = Path("data")
VIS_SAVE_PATH = Path("outputs/pair_debug/test_infer_pair.png")
PAIR_PROMPT = (
    "How many buildings have been changed in these two remote sensing images? "
    "Answer with only one integer."
)


transformers_logging.set_verbosity_error()


def suppress_noisy_logs() -> None:
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):    
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def parse_first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    return int(match.group(0))


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_image_path(raw_path: str, data_root: Path) -> Path:
    image_path = Path(raw_path)
    if image_path.exists():
        return image_path

    if image_path.is_absolute() and image_path.parts[1:2] == ("data",):
        image_path = data_root / Path(*image_path.parts[2:])
    elif not image_path.is_absolute():
        image_path = data_root / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {raw_path}")

    return image_path


def first_token_id(token_id) -> int | None:
    if isinstance(token_id, (list, tuple)):
        return token_id[0] if token_id else None
    return token_id


def configure_generation_tokens(model, processor) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is None:
        pad_token_id = first_token_id(eos_token_id)

    if pad_token_id is None:
        pad_token_id = first_token_id(
            getattr(model.generation_config, "eos_token_id", None)
        )

    if pad_token_id is not None:
        model.generation_config.pad_token_id = pad_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = pad_token_id


def load_record(annotation_path: Path, index: int) -> dict:
    with annotation_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if index < 0 or index >= len(records):
        raise IndexError(f"SAMPLE_INDEX {index} out of range (total={len(records)})")
    return records[index]


def normalize_record_question(record: dict) -> str:
    question = str(record["conversations"][0]["value"]).strip()
    question = question.replace("<image>", "").replace("  ", " ").strip()
    if "Answer with only one integer" not in question:
        question = f"{question} Answer with only one integer."
    return question


def build_prompt(processor, image_a: Image.Image, image_b: Image.Image, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_a},
                {"type": "image", "image": image_b},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def save_pair_figure(
    image_a: Image.Image,
    image_b: Image.Image,
    save_path: Path,
    sample_index: int,
    gt: int | None,
    pred: int | None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_a)
    axes[1].imshow(image_b)
    axes[0].set_title("Image A")
    axes[1].set_title("Image B")
    axes[0].axis("off")
    axes[1].axis("off")
    fig.suptitle(f"sample={sample_index} | GT={gt} | PRED={pred}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run one pair-image RS-VQA inference sample.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--annotation-path", type=Path, default=ANNOTATION_PATH)
    parser.add_argument("--data-root", type=Path, default=LOCAL_DATA_ROOT)
    parser.add_argument("--vis-save-path", type=Path, default=VIS_SAVE_PATH)
    parser.add_argument("--question", default=PAIR_PROMPT)
    parser.add_argument(
        "--use-record-question",
        action="store_true",
        help="Use the question stored in the annotation record instead of --question.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suppress_noisy_logs()

    annotation_path = repo_path(args.annotation_path)
    data_root = repo_path(args.data_root)
    vis_save_path = repo_path(args.vis_save_path)

    record = load_record(annotation_path, args.sample_index)
    gt_raw_answer = str(record["conversations"][1]["value"]).strip()
    gt = parse_first_int(gt_raw_answer)
    image_a_path = resolve_image_path(record["images"][0], data_root)
    image_b_path = resolve_image_path(record["images"][1], data_root)
    question = normalize_record_question(record) if args.use_record_question else args.question

    print(f"model_id: {args.model_id}")
    print(f"sample_index: {args.sample_index}")
    print(f"annotation_path: {annotation_path}")
    print(f"data_root: {data_root}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"prompt_question: {question}")
    print(f"gt_raw_answer: {gt_raw_answer}")
    print(f"gt(parsed): {gt}")
    print(f"vis_save_path: {vis_save_path}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        tie_word_embeddings=False,
        trust_remote_code=True,
    ).eval()
    configure_generation_tokens(model, processor)

    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")
    prompt = build_prompt(processor, image_a, image_b, question)

    try:
        inputs = processor(
            text=[prompt],
            images=[[image_a, image_b]],
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        inputs = processor(
            text=[prompt],
            images=[image_a, image_b],
            return_tensors="pt",
            padding=True,
        )

    inputs = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    input_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=model.generation_config.pad_token_id,
        )

    if input_length and generated_ids.shape[-1] > input_length:
        generated_ids = generated_ids[:, input_length:]

    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    pred = parse_first_int(raw_output)

    print("\n===== MODEL RAW OUTPUT =====")
    print(raw_output)
    print(f"pred(parsed): {pred}")

    save_pair_figure(
        image_a=image_a,
        image_b=image_b,
        save_path=vis_save_path,
        sample_index=args.sample_index,
        gt=gt,
        pred=pred,
    )
    print(f"saved visualization: {vis_save_path}")


if __name__ == "__main__":
    main()
