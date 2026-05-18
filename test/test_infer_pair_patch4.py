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


REPO_ROOT = Path(__file__).resolve().parents[1]


# Test-only defaults. All paths are resolved from the repository root.
MODEL_ID = "OpenGVLab/InternVL3_5-38B-HF"
SAMPLE_INDEX = 300
MAX_NEW_TOKENS = 128
ANNOTATION_PATH = Path(
    "data/coding/muti_task_data/test_task_data/count_build.json"
)
LOCAL_DATA_ROOT = Path("data")
VIS_SAVE_PATH = Path("outputs/patch_debug/test_infer_pair_patch4.png")

PATCH_GRID = 2  # 2x2 => 4 patches
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


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


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


def load_record(annotation_path: Path, index: int) -> dict:
    with annotation_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if index < 0 or index >= len(records):
        raise IndexError(f"SAMPLE_INDEX {index} out of range (total={len(records)})")
    return records[index]


def split_into_4_patches(image: Image.Image) -> list[tuple[int, tuple[int, int, int, int], Image.Image]]:
    width, height = image.size
    mid_x = width // PATCH_GRID
    mid_y = height // PATCH_GRID
    x_edges = [0, mid_x, width]
    y_edges = [0, mid_y, height]

    patches: list[tuple[int, tuple[int, int, int, int], Image.Image]] = []
    patch_id = 0
    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            left = x_edges[col]
            top = y_edges[row]
            right = x_edges[col + 1]
            bottom = y_edges[row + 1]
            box = (left, top, right, bottom)
            patches.append((patch_id, box, image.crop(box)))
            patch_id += 1
    return patches


def build_single_image_prompt(processor, image: Image.Image, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def infer_patch_count(
    model,
    processor,
    image_patch: Image.Image,
    question: str,
) -> tuple[str, int | None]:
    prompt = build_single_image_prompt(processor, image_patch, question)
    inputs = processor(
        text=[prompt],
        images=[image_patch],
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
    return raw_output, pred


def short_text(text: str, max_len: int = 60) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def save_patch_figure(
    patch_results: list[dict],
    gt: int | None,
    save_path: Path,
    sample_index: int,
) -> None:
    if not patch_results:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(patch_results)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    if rows == 1:
        axes = [axes]

    for row_idx, result in enumerate(patch_results):
        ax_a, ax_b = axes[row_idx]
        ax_a.imshow(result["patch_a"])
        ax_b.imshow(result["patch_b"])
        ax_a.axis("off")
        ax_b.axis("off")

        ax_a.set_title(
            f"A patch {result['patch_id']}\n"
            f"pred={result['pred_a']} | raw='{short_text(result['raw_a'])}'",
            fontsize=10,
        )
        ax_b.set_title(
            f"B patch {result['patch_id']}\n"
            f"pred={result['pred_b']} | diff(B-A)={result['diff']}",
            fontsize=10,
        )

    fig.suptitle(
        f"Patch-wise Building Count (sample={sample_index}, gt_changed={gt})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    suppress_noisy_logs()
    annotation_path = repo_path(ANNOTATION_PATH)
    data_root = repo_path(LOCAL_DATA_ROOT)
    vis_save_path = repo_path(VIS_SAVE_PATH)

    record = load_record(annotation_path, SAMPLE_INDEX)
    gt_raw_answer = str(record["conversations"][1]["value"]).strip()
    gt = parse_first_int(gt_raw_answer)

    image_a_path = resolve_image_path(record["images"][0], data_root)
    image_b_path = resolve_image_path(record["images"][1], data_root)
    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")

    print(f"model_id: {MODEL_ID}")
    print(f"sample_index: {SAMPLE_INDEX}")
    print(f"annotation_path: {annotation_path}")
    print(f"data_root: {data_root}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"prompt_question: {PATCH_PROMPT}")
    print(f"gt_changed_buildings(raw): {gt_raw_answer}")
    print(f"gt_changed_buildings(parsed): {gt}")
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

    patches_a = split_into_4_patches(image_a)
    patches_b = split_into_4_patches(image_b)

    print("\n===== PATCH-WISE BUILDING COUNT (A vs B) =====")
    sum_a = 0
    sum_b = 0
    sum_abs_diff = 0
    valid_pairs = 0
    patch_results: list[dict] = []

    for (patch_id_a, box_a, patch_a), (patch_id_b, box_b, patch_b) in zip(
        patches_a, patches_b
    ):
        if patch_id_a != patch_id_b:
            raise RuntimeError("Patch id mismatch between A and B.")

        raw_a, pred_a = infer_patch_count(
            model=model,
            processor=processor,
            image_patch=patch_a,
            question=PATCH_PROMPT,
        )
        raw_b, pred_b = infer_patch_count(
            model=model,
            processor=processor,
            image_patch=patch_b,
            question=PATCH_PROMPT,
        )

        diff = pred_b - pred_a if pred_a is not None and pred_b is not None else None
        abs_diff = abs(diff) if diff is not None else None

        if pred_a is not None:
            sum_a += pred_a
        if pred_b is not None:
            sum_b += pred_b
        if abs_diff is not None:
            sum_abs_diff += abs_diff
            valid_pairs += 1

        print(f"[patch {patch_id_a}] box_a={box_a}, box_b={box_b}")
        print(f"  A raw='{raw_a}' -> pred={pred_a}")
        print(f"  B raw='{raw_b}' -> pred={pred_b}")
        print(f"  diff(B-A)={diff}, abs_diff={abs_diff}")
        patch_results.append(
            {
                "patch_id": patch_id_a,
                "box_a": box_a,
                "box_b": box_b,
                "patch_a": patch_a,
                "patch_b": patch_b,
                "raw_a": raw_a,
                "raw_b": raw_b,
                "pred_a": pred_a,
                "pred_b": pred_b,
                "diff": diff,
                "abs_diff": abs_diff,
            }
        )

    print("\n===== SUMMARY =====")
    print(f"sum_count_A(valid): {sum_a}")
    print(f"sum_count_B(valid): {sum_b}")
    print(f"net_diff(B-A): {sum_b - sum_a}")
    print(f"sum_abs_diff_over_patches: {sum_abs_diff}")
    print(f"valid_patch_pairs: {valid_pairs}/{PATCH_GRID * PATCH_GRID}")
    print("note: patch-based aggregate is heuristic and may not exactly match GT change count.")

    save_patch_figure(
        patch_results=patch_results,
        gt=gt,
        save_path=vis_save_path,
        sample_index=SAMPLE_INDEX,
    )
    print(f"saved visualization: {vis_save_path}")


if __name__ == "__main__":
    main()
