from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    logging as transformers_logging,
)


# Edit these globals for a quick one-pair test.
MODEL_ID = "OpenGVLab/InternVL3_5-8B-HF"
SAMPLE_INDEX = 0
MAX_NEW_TOKENS = 128
USE_RAW_QUESTION_FOR_PROMPT = False
PROMPT_QUESTION = (
    "How many buildings have been changed in these two remote sensing images? "
    "Answer with only one integer."
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
ANNOTATION_PATH = (
    DATA_ROOT / "coding" / "muti_task_data" / "test_task_data" / "count_build.json"
)


transformers_logging.set_verbosity_error()


def suppress_noisy_logs() -> None:
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def resolve_image_path(path: str, data_root: Path) -> Path:
    image_path = Path(path)
    if image_path.exists():
        return image_path

    if image_path.is_absolute() and image_path.parts[1:2] == ("data",):
        image_path = data_root / Path(*image_path.parts[2:])
    elif not image_path.is_absolute():
        image_path = data_root / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return image_path


def parse_first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    return int(match.group(0))


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
        pad_token_id = first_token_id(getattr(model.generation_config, "eos_token_id", None))

    if pad_token_id is not None:
        model.generation_config.pad_token_id = pad_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = pad_token_id


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


def load_sample(index: int) -> dict:
    with ANNOTATION_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if index < 0 or index >= len(records):
        raise IndexError(f"SAMPLE_INDEX {index} is out of range. Total samples: {len(records)}")

    return records[index]


def main() -> None:
    suppress_noisy_logs()

    record = load_sample(SAMPLE_INDEX)
    raw_question = str(record["conversations"][0]["value"]).strip()
    raw_answer = str(record["conversations"][1]["value"]).strip()
    gt = parse_first_int(raw_answer)

    image_a_path = resolve_image_path(record["images"][0], DATA_ROOT)
    image_b_path = resolve_image_path(record["images"][1], DATA_ROOT)
    question_for_prompt = raw_question if USE_RAW_QUESTION_FOR_PROMPT else PROMPT_QUESTION

    print(f"sample_index: {SAMPLE_INDEX}")
    print(f"model_id: {MODEL_ID}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"raw_question: {raw_question}")
    print(f"prompt_question: {question_for_prompt}")
    print(f"raw_answer: {raw_answer}")
    print(f"gt: {gt}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        tie_word_embeddings=False,
        trust_remote_code=True,
    ).eval()
    configure_generation_tokens(model, processor)

    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")
    prompt = build_prompt(processor, image_a, image_b, question_for_prompt)

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
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=model.generation_config.pad_token_id,
        )

    if input_length and generated_ids.shape[-1] > input_length:
        generated_ids = generated_ids[:, input_length:]

    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    pred = parse_first_int(raw_output)

    print(f"raw_output: {raw_output}")
    print(f"pred: {pred}")
    print(f"correct: {pred == gt}")


if __name__ == "__main__":
    main()
