from __future__ import annotations

import json
import sys
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util import (
    build_prompt_with_images,
    configure_generation_tokens,
    move_to_model_device,
    parse_first_int,
    repo_path,
    resolve_image_path,
    suppress_http_logs,
)


# Test-only defaults. All paths are resolved from the repository root.
MODEL_ID = "OpenGVLab/InternVL3_5-1B-HF"
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
USE_RECORD_QUESTION = False


transformers_logging.set_verbosity_error()


def load_record(annotation_path: Path, index: int) -> dict:
    with annotation_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if index < 0 or index >= len(records):
        raise IndexError(f"SAMPLE_INDEX: {index} out of range (total={len(records)})")
    return records[index]


def normalize_record_question(record: dict) -> str:
    question = str(record["conversations"][0]["value"]).strip()
    question = question.replace("<image>", "").replace("  ", " ").strip()
    if "Answer with only one integer" not in question:
        question = f"{question} Answer with only one integer."
    return question


def save_pair_figure(
    image_a: Image.Image,
    image_b: Image.Image,
    save_path: Path,
    sample_index: int,
    gt: int | None,
    pred: int | None,
) -> tuple[Path, Path]:
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

    if save_path.suffix.lower() == ".svg":
        svg_path = save_path
        png_path = save_path.with_suffix(".png")
    else:
        png_path = save_path
        svg_path = save_path.with_suffix(".svg")

    fig.savefig(png_path, dpi=200)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    suppress_http_logs()

    annotation_path = repo_path(ANNOTATION_PATH)
    data_root = repo_path(LOCAL_DATA_ROOT)
    vis_save_path = repo_path(VIS_SAVE_PATH)

    record = load_record(annotation_path, SAMPLE_INDEX)
    gt_raw_answer = str(record["conversations"][1]["value"]).strip()
    gt = parse_first_int(gt_raw_answer)
    image_a_path = resolve_image_path(record["images"][0], data_root)
    image_b_path = resolve_image_path(record["images"][1], data_root)
    question = normalize_record_question(record) if USE_RECORD_QUESTION else PAIR_PROMPT

    print(f"model_id: {MODEL_ID}")
    print(f"sample_index: {SAMPLE_INDEX}")
    print(f"annotation_path: {annotation_path}")
    print(f"data_root: {data_root}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"prompt_question: {question}")
    print(f"gt_raw_answer: {gt_raw_answer}")
    print(f"gt(parsed): {gt}")
    print(f"vis_save_path: {vis_save_path}")

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
    prompt = build_prompt_with_images(processor, [image_a, image_b], question)

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

    inputs = move_to_model_device(inputs, model)
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

    png_path, svg_path = save_pair_figure(
        image_a=image_a,
        image_b=image_b,
        save_path=vis_save_path,
        sample_index=SAMPLE_INDEX,
        gt=gt,
        pred=pred,
    )
    print(f"saved visualization (png): {png_path}")
    print(f"saved visualization (svg): {svg_path}")


if __name__ == "__main__":
    main()
