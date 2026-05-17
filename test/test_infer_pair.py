from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    logging as transformers_logging,
)


# Test-only absolute paths/settings.
MODEL_ID = "OpenGVLab/InternVL3_5-1B-HF"
SAMPLE_INDEX = 300
MAX_NEW_TOKENS = 128
ANNOTATION_PATH = Path(
    "/Users/baeseojun/RS-VQA/data/coding/muti_task_data/test_task_data/count_build.json"
)
LOCAL_DATA_ROOT = Path("/Users/baeseojun/RS-VQA/data")
VIS_SAVE_PATH = Path("/Users/baeseojun/RS-VQA/outputs/pair_debug/test_infer_pair.png")
PATCH_PROMPT = (
    "How many buildings are visible in this remote sensing image? "
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


def resolve_image_path(raw_path: str) -> Path:
    image_path = Path(raw_path)
    if image_path.exists():
        return image_path

    # Dataset records often store paths like /data/...; remap to local absolute data root.
    if image_path.is_absolute() and image_path.parts[1:2] == ("data",):
        candidate = LOCAL_DATA_ROOT / Path(*image_path.parts[2:])
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found: {raw_path}")


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


def load_record(index: int) -> dict:
    with ANNOTATION_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if index < 0 or index >= len(records):
        raise IndexError(f"SAMPLE_INDEX {index} out of range (total={len(records)})")
    return records[index]


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


def main() -> None:
    suppress_noisy_logs()

    record = load_record(SAMPLE_INDEX)
    gt_raw_answer = str(record["conversations"][1]["value"]).strip()
    gt = parse_first_int(gt_raw_answer)
    image_a_path = resolve_image_path(record["images"][0])
    image_b_path = resolve_image_path(record["images"][1])

    print(f"model_id: {MODEL_ID}")
    print(f"sample_index: {SAMPLE_INDEX}")
    print(f"annotation_path: {ANNOTATION_PATH}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"prompt_question: {PATCH_PROMPT}")
    print(f"gt_raw_answer: {gt_raw_answer}")
    print(f"gt(parsed): {gt}")
    print(f"vis_save_path: {VIS_SAVE_PATH}")

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
    prompt = build_prompt(processor, image_a, image_b, PATCH_PROMPT)

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

    print("\n===== MODEL RAW OUTPUT =====")
    print(raw_output)
    print(f"pred(parsed): {pred}")

    save_pair_figure(
        image_a=image_a,
        image_b=image_b,
        save_path=VIS_SAVE_PATH,
        sample_index=SAMPLE_INDEX,
        gt=gt,
        pred=pred,
    )
    print(f"saved visualization: {VIS_SAVE_PATH}")


if __name__ == "__main__":
    main()
