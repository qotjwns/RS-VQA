from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    Component,
    build_prompt_with_images,
    connected_components,
    configure_generation_tokens,
    load_binary_mask,
    load_test_building_count,
    move_to_model_device,
    parse_first_int,
    repo_path,
    resolve_mask_paths,
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
# e.g., 2 -> 4 patches, 4 -> 16 patches.
PATCH_GRID = 2
PATCH_PROMPT = (
    "How many buildings are visible in this remote sensing image? "
    "Answer with only one integer."
)

MASK_CONNECTIVITY = 8
MIN_COMPONENT_AREA = 1
GT_POINT_COLOR = "#EF4444"
PRED_POINT_COLOR = "#FDE047"


transformers_logging.set_verbosity_error()


def patch_idx_from_centroid(
    centroid_x: float,
    centroid_y: float,
    width: int,
    height: int,
    patch_grid: int,
) -> int:
    col = min(int((centroid_x / width) * patch_grid), patch_grid - 1)
    row = min(int((centroid_y / height) * patch_grid), patch_grid - 1)
    return row * patch_grid + col


def build_patch_component_map(
    components: list[Component],
    width: int,
    height: int,
    patch_grid: int,
) -> tuple[list[list[Component]], list[int]]:
    patch_components: list[list[Component]] = [
        [] for _ in range(patch_grid * patch_grid)
    ]
    patch_gt_counts = [0 for _ in range(patch_grid * patch_grid)]

    for component in components:
        patch_idx = patch_idx_from_centroid(
            centroid_x=component.centroid_x,
            centroid_y=component.centroid_y,
            width=width,
            height=height,
            patch_grid=patch_grid,
        )
        patch_components[patch_idx].append(component)
        patch_gt_counts[patch_idx] += 1

    return patch_components, patch_gt_counts


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


def build_pred_points_in_patch(
    patch_components: list[Component],
    pred_count: int,
    box: tuple[int, int, int, int],
) -> list[tuple[float, float]]:
    if pred_count <= 0 or not patch_components:
        return []

    left, top, right, bottom = box
    sorted_components = sorted(patch_components, key=lambda row: row.area, reverse=True)
    total_components = len(sorted_components)
    points: list[tuple[float, float]] = []

    for index in range(pred_count):
        component = sorted_components[index % total_components]
        cycle = index // total_components
        if cycle == 0:
            dx = 0.0
            dy = 0.0
        else:
            angle = math.radians((index * 137.5) % 360.0)
            radius = 1.4 + cycle * 1.2
            dx = radius * math.cos(angle)
            dy = radius * math.sin(angle)

        x = min(max(component.centroid_x + dx, float(left)), float(right - 1))
        y = min(max(component.centroid_y + dy, float(top)), float(bottom - 1))
        points.append((x, y))

    return points


def full_to_patch_points(
    points: list[tuple[float, float]],
    box: tuple[int, int, int, int],
) -> list[tuple[float, float]]:
    left, top, _, _ = box
    return [(x - left, y - top) for x, y in points]


def save_patch_figure(
    patch_results: list[dict],
    gt: int | None,
    save_path: Path,
    sample_index: int,
    sum_diff_valid: int,
    sum_abs_diff: int,
    seg_total: int,
    pred_points_rendered: int,
    pred_points_unplaced: int,
) -> tuple[Path, Path]:
    if not patch_results:
        return save_path.with_suffix(".png"), save_path.with_suffix(".svg")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(patch_results)
    fig, axes = plt.subplots(rows, 2, figsize=(11, 4.2 * rows))
    if rows == 1:
        axes = [axes]

    for row_idx, result in enumerate(patch_results):
        ax_a, ax_b = axes[row_idx]
        ax_a.imshow(result["patch_a"])
        ax_b.imshow(result["patch_b"])
        ax_a.axis("off")
        ax_b.axis("off")

        ax_a.set_title(
            f"A patch {result['patch_id'] + 1} | pred={result['pred_a']}",
            fontsize=10,
        )

        pred_points = result["pred_points_local"]
        if pred_points:
            ax_b.scatter(
                [x for x, _ in pred_points],
                [y for _, y in pred_points],
                c=PRED_POINT_COLOR,
                s=28,
                marker="o",
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
                zorder=2,
            )

        gt_points = result["gt_points_local"]
        if gt_points:
            ax_b.scatter(
                [x for x, _ in gt_points],
                [y for _, y in gt_points],
                c=GT_POINT_COLOR,
                s=52,
                marker="x",
                linewidths=1.5,
                alpha=0.98,
                zorder=4,
            )

        ax_b.set_title(
            (
                f"B patch {result['patch_id'] + 1} | pred={result['pred_b']} | "
                f"diff={result['diff']} | |diff|={result['abs_diff']}\n"
                f"GT(mask)={result['gt_patch']} | GT(red X)={len(gt_points)} | Pred(yellow)={len(pred_points)}"
            ),
            fontsize=9,
        )

    fig.suptitle(
        (
            f"Patch-wise Count + Predicted Points (sample={sample_index}, gt_answer={gt})\n"
            f"sum_diff_valid={sum_diff_valid}, sum_abs_diff={sum_abs_diff}, "
            f"seg_total={seg_total}, rendered_points={pred_points_rendered}, "
            f"unplaced_points={pred_points_unplaced}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

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
    gt = parse_first_int(gt_raw_answer)

    image_a_path = Path(sample.image_a)
    image_b_path = Path(sample.image_b)
    label_path, label_rgb_path = resolve_mask_paths(image_a_path)
    if label_path is None and label_rgb_path is None:
        raise FileNotFoundError(f"No label/label_rgb found for: {image_a_path}")

    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")

    binary_mask, mask_source = load_binary_mask(
        label_path=label_path,
        label_rgb_path=label_rgb_path,
    )
    all_components = connected_components(binary=binary_mask, connectivity=MASK_CONNECTIVITY)
    components = [row for row in all_components if row.area >= MIN_COMPONENT_AREA]
    patch_components, patch_gt_counts = build_patch_component_map(
        components=components,
        width=binary_mask.shape[1],
        height=binary_mask.shape[0],
        patch_grid=PATCH_GRID,
    )
    seg_total = int(sum(patch_gt_counts))

    print(f"model_id: {MODEL_ID}")
    print(f"sample_index: {SAMPLE_INDEX}")
    print(f"annotation_path: {annotation_path}")
    print(f"data_root: {data_root}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"mask_path(label): {label_path}")
    print(f"mask_path(label_rgb): {label_rgb_path}")
    print(f"mask_source: {mask_source}")
    print(f"prompt_question: {PATCH_PROMPT}")
    print(f"gt_changed_buildings(raw): {gt_raw_answer}")
    print(f"gt_changed_buildings(parsed): {gt}")
    print(f"vis_save_path: {vis_save_path}")
    print(f"patch_grid: {PATCH_GRID}x{PATCH_GRID}")
    print(f"mask_connectivity: {MASK_CONNECTIVITY}")
    print(f"min_component_area: {MIN_COMPONENT_AREA}")

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

    patches_a = split_into_grid_patches(image_a, PATCH_GRID)
    patches_b = split_into_grid_patches(image_b, PATCH_GRID)

    print("\n===== PATCH-WISE BUILDING COUNT (A vs B) =====")
    sum_a = 0
    sum_b = 0
    sum_abs_diff = 0
    sum_diff_valid = 0
    valid_pairs = 0
    pred_points_rendered = 0
    pred_points_unplaced = 0
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
        if diff is not None:
            sum_diff_valid += diff
            sum_abs_diff += abs_diff if abs_diff is not None else 0
            valid_pairs += 1

        pred_changed = abs_diff if abs_diff is not None else 0
        patch_pred_points_full = build_pred_points_in_patch(
            patch_components=patch_components[patch_id_a],
            pred_count=pred_changed,
            box=box_b,
        )
        patch_pred_points_local = full_to_patch_points(
            points=patch_pred_points_full,
            box=box_b,
        )
        patch_gt_points_local = full_to_patch_points(
            points=[
                (component.centroid_x, component.centroid_y)
                for component in patch_components[patch_id_a]
            ],
            box=box_b,
        )

        pred_points_rendered += len(patch_pred_points_local)
        pred_points_unplaced += max(0, pred_changed - len(patch_pred_points_local))

        print(f"[patch {patch_id_a}] box_a={box_a}, box_b={box_b}")
        print(f"  A raw='{raw_a}' -> pred={pred_a}")
        print(f"  B raw='{raw_b}' -> pred={pred_b}")
        print(f"  diff(B-A)={diff}, abs_diff={abs_diff}, gt_patch={patch_gt_counts[patch_id_a]}")
        print(
            f"  pred_points(rendered/unplaced)="
            f"{len(patch_pred_points_local)}/{max(0, pred_changed - len(patch_pred_points_local))}"
        )
        print(f"  gt_points(mask-centroids)={len(patch_gt_points_local)}")

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
                "gt_patch": patch_gt_counts[patch_id_a],
                "gt_points_local": patch_gt_points_local,
                "pred_points_local": patch_pred_points_local,
            }
        )

    print("\n===== SUMMARY =====")
    print(f"sum_count_A(valid): {sum_a}")
    print(f"sum_count_B(valid): {sum_b}")
    print(f"sum_diff_valid(B-A): {sum_diff_valid}")
    print(f"sum_abs_diff_over_patches: {sum_abs_diff}")
    print(f"seg_total(mask-derived): {seg_total}")
    print(f"valid_patch_pairs: {valid_pairs}/{PATCH_GRID * PATCH_GRID}")
    print(f"pred_points_rendered: {pred_points_rendered}")
    print(f"pred_points_unplaced: {pred_points_unplaced}")
    print("note: diff method is kept as signed sum (sum_diff_valid).")

    png_path, svg_path = save_patch_figure(
        patch_results=patch_results,
        gt=gt,
        save_path=vis_save_path,
        sample_index=SAMPLE_INDEX,
        sum_diff_valid=sum_diff_valid,
        sum_abs_diff=sum_abs_diff,
        seg_total=seg_total,
        pred_points_rendered=pred_points_rendered,
        pred_points_unplaced=pred_points_unplaced,
    )
    print(f"saved visualization (png): {png_path}")
    print(f"saved visualization (svg): {svg_path}")


if __name__ == "__main__":
    main()
