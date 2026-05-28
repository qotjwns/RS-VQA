from __future__ import annotations

import sys
import textwrap
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
    load_test_building_count,
    move_to_model_device,
    parse_first_int,
    repo_path,
    suppress_http_logs,
)


# Test-only defaults. All paths are resolved from the repository root.
MODEL_ID = "OpenGVLab/InternVL3_5-1B-HF"
SAMPLE_INDEX = 300
MAX_NEW_TOKENS = 128
ANNOTATION_PATH = Path("data/coding/muti_task_data/test_task_data/count_build.json")
LOCAL_DATA_ROOT = Path("data")
VIS_SAVE_PATH = Path("outputs/patch_debug/test_infer_pair_patch4.png")

# Number of splits per side. Total patches = PATCH_GRID * PATCH_GRID.
PATCH_GRID = 2
PATCH_PROMPT = (
    "How many buildings are visible in this remote sensing image? "
    "Answer with only one integer."
)


transformers_logging.set_verbosity_error()


def split_into_grid_patches(
    image: Image.Image,
    patch_grid: int,
) -> list[tuple[int, tuple[int, int, int, int], Image.Image]]:
    if patch_grid < 1:
        raise ValueError(f"PATCH_GRID must be >= 1, got {patch_grid}")

    width, height = image.size
    x_edges = [int(width * index / patch_grid) for index in range(patch_grid + 1)]
    y_edges = [int(height * index / patch_grid) for index in range(patch_grid + 1)]

    patches: list[tuple[int, tuple[int, int, int, int], Image.Image]] = []
    patch_id = 0
    for row in range(patch_grid):
        for col in range(patch_grid):
            left = x_edges[col]
            top = y_edges[row]
            right = x_edges[col + 1]
            bottom = y_edges[row + 1]
            box = (left, top, right, bottom)
            patches.append((patch_id, box, image.crop(box)))
            patch_id += 1
    return patches


def infer_patch_count(
    model,
    processor,
    image_patch: Image.Image,
    question: str,
) -> tuple[str, int | None]:
    prompt = build_prompt_with_images(processor, [image_patch], question)
    inputs = processor(
        text=[prompt],
        images=[image_patch],
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
    if pred is not None and pred < 0:
        pred = None
    return raw_output, pred


def save_patch_figure(
    patch_results: list[dict],
    save_path: Path,
    sample_index: int,
    gt_answer: int | None,
    sum_diff_valid: int,
    sum_abs_diff: int,
) -> tuple[Path, Path]:
    if not patch_results:
        return save_path.with_suffix(".png"), save_path.with_suffix(".svg")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(patch_results)
    fig, axes = plt.subplots(rows, 3, figsize=(15.5, 4.0 * rows))

    if rows == 1:
        axes = [axes]

    for row_idx, result in enumerate(patch_results):
        ax_a, ax_b, ax_text = axes[row_idx]
        ax_a.imshow(result["patch_a"])
        ax_b.imshow(result["patch_b"])
        ax_a.axis("off")
        ax_b.axis("off")

        ax_a.set_title(f"A patch {result['patch_id'] + 1} | pred={result['pred_a']}", fontsize=10)
        ax_b.set_title(f"B patch {result['patch_id'] + 1} | pred={result['pred_b']}", fontsize=10)

        ax_text.axis("off")
        ax_text.set_title("Patch Output", fontsize=10)
        raw_a = result["raw_a"] if result["raw_a"] else "(empty)"
        raw_b = result["raw_b"] if result["raw_b"] else "(empty)"
        message = (
            f"diff(B-A): {result['diff']}\n"
            f"|diff|: {result['abs_diff']}\n\n"
            f"raw_a:\n{textwrap.fill(raw_a, width=36)}\n\n"
            f"raw_b:\n{textwrap.fill(raw_b, width=36)}"
        )
        ax_text.text(0.0, 1.0, message, va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle(
        (
            f"Patch-wise Inference (sample={sample_index}, gt_answer={gt_answer}) | "
            f"sum_diff_valid={sum_diff_valid}, sum_abs_diff={sum_abs_diff}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path.suffix.lower() == ".svg":
        svg_path = save_path
        png_path = save_path.with_suffix(".png")
    else:
        png_path = save_path
        svg_path = save_path.with_suffix(".svg")

    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    suppress_http_logs()
    annotation_path = repo_path(ANNOTATION_PATH)
    data_root = repo_path(LOCAL_DATA_ROOT)
    vis_save_path = repo_path(VIS_SAVE_PATH)

    samples = load_test_building_count(
        annotation_path=annotation_path,
        data_root=data_root,
    )
    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= len(samples):
        raise IndexError(f"SAMPLE_INDEX {SAMPLE_INDEX} out of range (total={len(samples)})")

    sample = samples[SAMPLE_INDEX]
    gt_raw_answer = str(sample.answer).strip()
    gt_answer = parse_first_int(gt_raw_answer)
    image_a_path = Path(sample.image_a)
    image_b_path = Path(sample.image_b)

    print(f"model_id: {MODEL_ID}")
    print(f"sample_index: {SAMPLE_INDEX}")
    print(f"annotation_path: {annotation_path}")
    print(f"data_root: {data_root}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"prompt_question: {PATCH_PROMPT}")
    print(f"gt_changed_buildings(raw): {gt_raw_answer}")
    print(f"gt_changed_buildings(parsed): {gt_answer}")
    print(f"vis_save_path: {vis_save_path}")
    print(f"patch_grid: {PATCH_GRID}x{PATCH_GRID}")

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
    patches_a = split_into_grid_patches(image_a, PATCH_GRID)
    patches_b = split_into_grid_patches(image_b, PATCH_GRID)

    if len(patches_a) != len(patches_b):
        raise RuntimeError("Patch count mismatch between A and B")

    print("\n===== PATCH-WISE INFERENCE (A vs B) =====")
    sum_a = 0
    sum_b = 0
    sum_abs_diff = 0
    sum_diff_valid = 0
    valid_pairs = 0
    patch_results: list[dict] = []

    for (patch_id_a, box_a, patch_a), (patch_id_b, box_b, patch_b) in zip(patches_a, patches_b):
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
        if diff is not None:
            sum_diff_valid += diff
            sum_abs_diff += abs_diff if abs_diff is not None else 0
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
    print(f"sum_diff_valid(B-A): {sum_diff_valid}")
    print(f"sum_abs_diff_over_patches: {sum_abs_diff}")
    print(f"valid_patch_pairs: {valid_pairs}/{PATCH_GRID * PATCH_GRID}")

    png_path, svg_path = save_patch_figure(
        patch_results=patch_results,
        save_path=vis_save_path,
        sample_index=SAMPLE_INDEX,
        gt_answer=gt_answer,
        sum_diff_valid=sum_diff_valid,
        sum_abs_diff=sum_abs_diff,
    )
    print(f"saved visualization (png): {png_path}")
    print(f"saved visualization (svg): {svg_path}")


if __name__ == "__main__":
    main()
