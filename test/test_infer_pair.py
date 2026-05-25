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
VIS_SAVE_PATH = Path("outputs/pair_debug/test_infer_pair.png")
PAIR_PROMPT = (
    "How many buildings have been changed in these two remote sensing images? "
    "Answer with only one integer."
)
MASK_CONNECTIVITY = 8
MIN_COMPONENT_AREA = 1

GT_POINT_COLOR = "#EF4444"
PRED_POINT_COLOR = "#FDE047"


transformers_logging.set_verbosity_error()


def infer_pair_count_baseline_style(
    model,
    processor,
    image_a: Image.Image,
    image_b: Image.Image,
    question: str,
) -> tuple[str, int | None]:
    prompt = build_prompt_with_images(processor, [image_a, image_b], question)
    inputs = processor(
        text=[prompt],
        images=[[image_a, image_b]],
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


def build_points_from_components(
    components: list[Component],
    count: int | None,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    if count is None or count <= 0 or not components:
        return []

    sorted_components = sorted(components, key=lambda row: row.area, reverse=True)
    points: list[tuple[float, float]] = []
    total_components = len(sorted_components)

    for index in range(count):
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

        x = min(max(component.centroid_x + dx, 0.0), float(width - 1))
        y = min(max(component.centroid_y + dy, 0.0), float(height - 1))
        points.append((x, y))

    return points


def save_pair_figure(
    image_a: Image.Image,
    image_b: Image.Image,
    binary_mask: np.ndarray,
    gt_points: list[tuple[float, float]],
    pred_points: list[tuple[float, float]],
    save_path: Path,
    sample_index: int,
    gt_answer: int | None,
    pred_answer: int | None,
    seg_total: int,
) -> tuple[Path, Path]:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_a_np = np.array(image_a)
    image_b_np = np.array(image_b)
    h, w = binary_mask.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_a, ax_b, ax_overlay = axes

    ax_a.imshow(image_a_np)
    ax_a.set_title("T1 (Image A)")
    ax_a.axis("off")

    ax_b.imshow(image_b_np)
    ax_b.set_title("T2 (Image B)")
    ax_b.axis("off")

    ax_overlay.imshow(image_b_np)
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.12
    overlay[..., 2] = 0.05
    overlay[..., 3] = binary_mask.astype(np.float32) * 0.38
    ax_overlay.imshow(overlay)

    gt_artist = None
    pred_artist = None

    if pred_points:
        pred_artist = ax_overlay.scatter(
            [x for x, _ in pred_points],
            [y for _, y in pred_points],
            c=PRED_POINT_COLOR,
            s=28,
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            zorder=2,
            label="Pred (yellow)",
        )

    if gt_points:
        gt_artist = ax_overlay.scatter(
            [x for x, _ in gt_points],
            [y for _, y in gt_points],
            c=GT_POINT_COLOR,
            s=52,
            marker="x",
            linewidths=1.5,
            alpha=0.98,
            zorder=4,
            label="GT (red X)",
        )

    legend_items = [item for item in (gt_artist, pred_artist) if item is not None]
    if legend_items:
        ax_overlay.legend(loc="lower right", fontsize=9, framealpha=0.8)

    ax_overlay.set_title("GT mask overlay + GT(red X) / Pred(yellow)")
    ax_overlay.axis("off")

    abs_error = (
        abs(pred_answer - gt_answer)
        if pred_answer is not None and gt_answer is not None
        else None
    )
    fig.suptitle(
        (
            f"sample={sample_index} | gt_answer={gt_answer} | pred_answer={pred_answer} | "
            f"abs_error={abs_error} | seg_total(mask-derived)={seg_total}"
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
    gt_answer = parse_first_int(gt_raw_answer)
    image_a_path = Path(sample.image_a)
    image_b_path = Path(sample.image_b)
    label_path, label_rgb_path = resolve_mask_paths(image_a_path)
    if label_path is None and label_rgb_path is None:
        raise FileNotFoundError(f"No label/label_rgb found for: {image_a_path}")

    print(f"model_id: {MODEL_ID}")
    print(f"sample_index: {SAMPLE_INDEX}")
    print(f"annotation_path: {annotation_path}")
    print(f"data_root: {data_root}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"mask_path(label): {label_path}")
    print(f"mask_path(label_rgb): {label_rgb_path}")
    print(f"prompt_question: {PAIR_PROMPT}")
    print(f"gt_raw_answer: {gt_raw_answer}")
    print(f"gt(parsed): {gt_answer}")
    print(f"vis_save_path: {vis_save_path}")
    print(f"mask_connectivity: {MASK_CONNECTIVITY}")
    print(f"min_component_area: {MIN_COMPONENT_AREA}")
    print("pair_input_mode: strict (images=[[image_a, image_b]])")

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

    binary_mask, mask_source = load_binary_mask(
        label_path=label_path,
        label_rgb_path=label_rgb_path,
    )
    print(f"mask_source: {mask_source}")

    components_all = connected_components(binary=binary_mask, connectivity=MASK_CONNECTIVITY)
    components = [row for row in components_all if row.area >= MIN_COMPONENT_AREA]
    seg_total = len(components)
    gt_points = [(row.centroid_x, row.centroid_y) for row in components]

    raw_output, pred_answer = infer_pair_count_baseline_style(
        model=model,
        processor=processor,
        image_a=image_a,
        image_b=image_b,
        question=PAIR_PROMPT,
    )

    pred_points = build_points_from_components(
        components=components,
        count=pred_answer,
        width=binary_mask.shape[1],
        height=binary_mask.shape[0],
    )

    print("\n===== SEGMENTATION-DERIVED GT =====")
    print(f"gt_points(mask components): {len(gt_points)}")
    print(f"seg_total(derived): {seg_total}")

    print("\n===== BASELINE-STYLE MODEL OUTPUT =====")
    print(raw_output)
    print(f"pred(parsed): {pred_answer}")
    print(f"pred_points(rendered): {len(pred_points)}")

    png_path, svg_path = save_pair_figure(
        image_a=image_a,
        image_b=image_b,
        binary_mask=binary_mask,
        gt_points=gt_points,
        pred_points=pred_points,
        save_path=vis_save_path,
        sample_index=SAMPLE_INDEX,
        gt_answer=gt_answer,
        pred_answer=pred_answer,
        seg_total=seg_total,
    )
    print(f"saved visualization (png): {png_path}")
    print(f"saved visualization (svg): {svg_path}")


if __name__ == "__main__":
    main()
