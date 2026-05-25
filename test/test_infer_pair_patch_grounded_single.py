from __future__ import annotations

import json
import re
import sys
import textwrap
from dataclasses import dataclass
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
    build_prompt_with_images,
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
PATCH_GRID = 2
MAX_NEW_TOKENS = 512
ANNOTATION_PATH = Path("data/coding/muti_task_data/test_task_data/count_build.json")
LOCAL_DATA_ROOT = Path("data")
VIS_SAVE_PATH = Path("outputs/grounded_count/test_infer_pair_patch_test_grounded.png")
MASK_CONNECTIVITY = 8
MIN_COMPONENT_AREA = 1
CENTER_HIT_MARGIN_PX = 2.0

PROMPT_TEMPLATE = (
    "Given two remote sensing patches (T1 then T2), count changed buildings and localize them in T2 patch. "
    "Return JSON only with keys count and points. "
    "Format exactly: {{\"count\": <int>, \"points\": [[x, y], ...]}}. "
    "Coordinates must be pixel integers in T2 patch space with 0 <= x < {width}, 0 <= y < {height}. "
    "The number of points must equal count."
)

PRED_POINT_COLOR = "#FDE047"
GT_BOX_COLOR = "#EF4444"


transformers_logging.set_verbosity_error()


@dataclass(frozen=True)
class ComponentBox:
    area: int
    x1: int
    y1: int
    x2: int
    y2: int


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


def connected_components_with_boxes(
    binary: np.ndarray,
    connectivity: int = 8,
) -> list[ComponentBox]:
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(binary)
    components: list[ComponentBox] = []

    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue

        stack = [(y, x)]
        visited[y, x] = True
        area = 0
        min_x = max_x = x
        min_y = max_y = y

        while stack:
            cy, cx = stack.pop()
            area += 1
            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy

            for dy, dx in offsets:
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    continue
                if visited[ny, nx] or not binary[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        components.append(
            ComponentBox(
                area=area,
                x1=min_x,
                y1=min_y,
                x2=max_x + 1,
                y2=max_y + 1,
            )
        )

    return components


def infer_grounded_patch_pair(
    model,
    processor,
    patch_a: Image.Image,
    patch_b: Image.Image,
    question: str,
) -> str:
    prompt = build_prompt_with_images(processor, [patch_a, patch_b], question)
    inputs = processor(
        text=[prompt],
        images=[[patch_a, patch_b]],
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

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def _extract_json_object(raw_text: str) -> dict | None:
    raw_text = raw_text.strip()
    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _normalize_box(
    box_candidate,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    values: list[float] = []

    if isinstance(box_candidate, dict):
        keys = ("x1", "y1", "x2", "y2")
        if not all(key in box_candidate for key in keys):
            return None
        try:
            values = [float(box_candidate[key]) for key in keys]
        except (TypeError, ValueError):
            return None
    elif isinstance(box_candidate, (list, tuple)) and len(box_candidate) == 4:
        try:
            values = [float(row) for row in box_candidate]
        except (TypeError, ValueError):
            return None
    else:
        return None

    x1, y1, x2, y2 = [int(round(row)) for row in values]
    x1 = min(max(x1, 0), width - 1)
    y1 = min(max(y1, 0), height - 1)
    x2 = min(max(x2, 0), width)
    y2 = min(max(y2, 0), height)

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    if x1 >= x2 or y1 >= y2:
        return None

    return (x1, y1, x2, y2)


def _normalize_point(
    point_candidate,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    values: list[float] = []
    if isinstance(point_candidate, dict):
        if "x" not in point_candidate or "y" not in point_candidate:
            return None
        try:
            values = [float(point_candidate["x"]), float(point_candidate["y"])]
        except (TypeError, ValueError):
            return None
    elif isinstance(point_candidate, (list, tuple)) and len(point_candidate) == 2:
        try:
            values = [float(point_candidate[0]), float(point_candidate[1])]
        except (TypeError, ValueError):
            return None
    else:
        return None

    x, y = [int(round(row)) for row in values]
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    return (x, y)


def _extract_box_quads_from_text(
    raw_text: str,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    pattern = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    boxes: list[tuple[int, int, int, int]] = []
    for match in pattern.finditer(raw_text):
        candidate = [match.group(1), match.group(2), match.group(3), match.group(4)]
        normalized = _normalize_box(candidate, width=width, height=height)
        if normalized is not None:
            boxes.append(normalized)
    return boxes


def _extract_point_pairs_from_text(
    raw_text: str,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    pattern = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    points: list[tuple[int, int]] = []
    for match in pattern.finditer(raw_text):
        candidate = [match.group(1), match.group(2)]
        normalized = _normalize_point(candidate, width=width, height=height)
        if normalized is not None:
            points.append(normalized)
    return points


def parse_grounded_output(
    raw_output: str,
    width: int,
    height: int,
) -> tuple[int | None, list[tuple[int, int]]]:
    parsed_json = _extract_json_object(raw_output)
    pred_count = parse_first_int(raw_output)

    points: list[tuple[int, int]] = []
    if parsed_json is None:
        points = _extract_point_pairs_from_text(raw_output, width=width, height=height)
        if not points:
            boxes = _extract_box_quads_from_text(raw_output, width=width, height=height)
            points = [box_center(box) for box in boxes]
    else:
        count_raw = parsed_json.get("count")
        try:
            pred_count = int(count_raw)
        except (TypeError, ValueError):
            pred_count = parse_first_int(raw_output)

        for point_candidate in parsed_json.get("points", []):
            normalized_point = _normalize_point(point_candidate, width=width, height=height)
            if normalized_point is not None:
                points.append(normalized_point)

        if not points:
            for box_candidate in parsed_json.get("boxes", []):
                normalized_box = _normalize_box(box_candidate, width=width, height=height)
                if normalized_box is not None:
                    points.append(box_center(normalized_box))

    if pred_count is not None and pred_count >= 0 and len(points) > pred_count:
        points = points[:pred_count]

    return pred_count, points


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2 - 1) * 0.5, (y1 + y2 - 1) * 0.5)


def center_in_expanded_box(
    cx: float,
    cy: float,
    box: tuple[int, int, int, int],
    margin_px: float,
) -> bool:
    x1, y1, x2, y2 = box
    return (
        cx >= (x1 - margin_px)
        and cx <= (x2 - 1 + margin_px)
        and cy >= (y1 - margin_px)
        and cy <= (y2 - 1 + margin_px)
    )


def greedy_center_hit_match(
    pred_points: list[tuple[int, int]],
    gt_boxes: list[tuple[int, int, int, int]],
    margin_px: float,
) -> tuple[list[tuple[int, int, float]], int, int, int]:
    # candidates: (distance_between_pred_point_and_gt_center, pred_idx, gt_idx)
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, (pred_x, pred_y) in enumerate(pred_points):
        for gt_idx, gt_box in enumerate(gt_boxes):
            if not center_in_expanded_box(pred_x, pred_y, gt_box, margin_px):
                continue
            gt_cx, gt_cy = box_center(gt_box)
            distance = ((pred_x - gt_cx) ** 2 + (pred_y - gt_cy) ** 2) ** 0.5
            candidates.append((distance, pred_idx, gt_idx))

    candidates.sort(key=lambda row: row[0])

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for distance, pred_idx, gt_idx in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, distance))

    tp = len(matches)
    fp = max(0, len(pred_points) - tp)
    fn = max(0, len(gt_boxes) - tp)
    return matches, tp, fp, fn


def draw_boxes(ax, boxes: list[tuple[int, int, int, int]], color: str, label_prefix: str) -> None:
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 3),
            f"{label_prefix}{idx}",
            color=color,
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1, "edgecolor": "none"},
        )


def draw_points(ax, points: list[tuple[int, int]], color: str, label_prefix: str) -> None:
    if not points:
        return
    ax.scatter(
        [x for x, _ in points],
        [y for _, y in points],
        c=color,
        s=26,
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        alpha=0.95,
        zorder=5,
    )
    for idx, (x, y) in enumerate(points, start=1):
        ax.text(
            x + 1,
            max(0, y - 2),
            f"{label_prefix}{idx}",
            color=color,
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 1, "edgecolor": "none"},
        )


def map_component_to_patch(
    component: ComponentBox,
    image_width: int,
    image_height: int,
    patch_grid: int,
) -> int:
    cx = (component.x1 + component.x2 - 1) / 2.0
    cy = (component.y1 + component.y2 - 1) / 2.0
    col = min(int((cx / image_width) * patch_grid), patch_grid - 1)
    row = min(int((cy / image_height) * patch_grid), patch_grid - 1)
    return row * patch_grid + col


def full_box_to_local_clipped(
    full_box: tuple[int, int, int, int],
    patch_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    fx1, fy1, fx2, fy2 = full_box
    px1, py1, px2, py2 = patch_box

    ix1 = max(fx1, px1)
    iy1 = max(fy1, py1)
    ix2 = min(fx2, px2)
    iy2 = min(fy2, py2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    return (ix1 - px1, iy1 - py1, ix2 - px1, iy2 - py1)


def build_patch_gt_boxes(
    components: list[ComponentBox],
    patches_b: list[tuple[int, tuple[int, int, int, int], Image.Image]],
    image_width: int,
    image_height: int,
    patch_grid: int,
) -> list[list[tuple[int, int, int, int]]]:
    patch_gt_boxes: list[list[tuple[int, int, int, int]]] = [
        [] for _ in range(patch_grid * patch_grid)
    ]

    patch_box_by_id = {patch_id: box for patch_id, box, _ in patches_b}

    for component in components:
        patch_id = map_component_to_patch(
            component=component,
            image_width=image_width,
            image_height=image_height,
            patch_grid=patch_grid,
        )
        patch_box = patch_box_by_id[patch_id]
        local_box = full_box_to_local_clipped(
            full_box=(component.x1, component.y1, component.x2, component.y2),
            patch_box=patch_box,
        )
        if local_box is not None:
            patch_gt_boxes[patch_id].append(local_box)

    return patch_gt_boxes


def save_patch_grounded_figure(
    patch_results: list[dict],
    save_path: Path,
    sample_index: int,
    gt_answer: int | None,
    total_tp: int,
    total_fp: int,
    total_fn: int,
    precision: float,
    recall: float,
) -> tuple[Path, Path]:
    if not patch_results:
        return save_path.with_suffix(".png"), save_path.with_suffix(".svg")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows = len(patch_results)
    fig, axes = plt.subplots(rows, 3, figsize=(16, 4.0 * rows))
    if rows == 1:
        axes = [axes]

    for row_idx, result in enumerate(patch_results):
        ax_a, ax_b, ax_text = axes[row_idx]
        ax_a.imshow(result["patch_a"])
        ax_b.imshow(result["patch_b"])
        ax_a.axis("off")
        ax_b.axis("off")

        ax_a.set_title(f"A patch {result['patch_id'] + 1}", fontsize=10)
        ax_b.set_title(
            (
                f"B patch {result['patch_id'] + 1} | pred_count={result['pred_count']}\n"
                f"tp/fp/fn={result['tp']}/{result['fp']}/{result['fn']}"
            ),
            fontsize=9,
        )

        draw_boxes(ax_b, result["gt_boxes"], GT_BOX_COLOR, "G")
        draw_points(ax_b, result["pred_points"], PRED_POINT_COLOR, "P")

        ax_text.axis("off")
        ax_text.set_title("Patch Output", fontsize=10)
        raw_output = result["raw_output"] if result["raw_output"] else "(empty output)"
        ax_text.text(
            0.0,
            1.0,
            (
                f"pred_count(parsed): {result['pred_count']}\n"
                f"gt_boxes: {len(result['gt_boxes'])}\n"
                f"pred_points: {len(result['pred_points'])}\n\n"
                f"raw_output:\n{textwrap.fill(raw_output, width=36)}"
            ),
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

    fig.suptitle(
        (
            f"Patch Grounded Counting (center-hit, sample={sample_index}, gt_answer={gt_answer}) | "
            f"TP={total_tp}, FP={total_fp}, FN={total_fn}, "
            f"precision={precision:.3f}, recall={recall:.3f}"
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
    label_path, label_rgb_path = resolve_mask_paths(image_a_path)
    if label_path is None and label_rgb_path is None:
        raise FileNotFoundError(f"No label/label_rgb found for: {image_a_path}")

    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")
    image_width, image_height = image_b.size

    print(f"model_id: {MODEL_ID}")
    print(f"sample_index: {SAMPLE_INDEX}")
    print(f"annotation_path: {annotation_path}")
    print(f"data_root: {data_root}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"mask_path(label): {label_path}")
    print(f"mask_path(label_rgb): {label_rgb_path}")
    print(f"gt_changed_buildings(raw): {gt_raw_answer}")
    print(f"gt_changed_buildings(parsed): {gt_answer}")
    print(f"vis_save_path: {vis_save_path}")
    print(f"patch_grid: {PATCH_GRID}x{PATCH_GRID}")
    print(f"center_hit_margin_px: {CENTER_HIT_MARGIN_PX}")

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

    binary_mask, mask_source = load_binary_mask(
        label_path=label_path,
        label_rgb_path=label_rgb_path,
    )
    print(f"mask_source: {mask_source}")

    components = connected_components_with_boxes(binary=binary_mask, connectivity=MASK_CONNECTIVITY)
    components = [row for row in components if row.area >= MIN_COMPONENT_AREA]

    patches_a = split_into_grid_patches(image_a, PATCH_GRID)
    patches_b = split_into_grid_patches(image_b, PATCH_GRID)
    if len(patches_a) != len(patches_b):
        raise RuntimeError("Patch count mismatch between A and B")

    patch_gt_boxes = build_patch_gt_boxes(
        components=components,
        patches_b=patches_b,
        image_width=image_width,
        image_height=image_height,
        patch_grid=PATCH_GRID,
    )

    print("\n===== PATCH GROUNDED COUNTING (A_i, B_i, center-hit) =====")
    patch_results: list[dict] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_count = 0

    for (patch_id_a, box_a, patch_a), (patch_id_b, box_b, patch_b) in zip(patches_a, patches_b):
        if patch_id_a != patch_id_b:
            raise RuntimeError("Patch id mismatch between A and B")

        patch_w, patch_h = patch_b.size
        question = PROMPT_TEMPLATE.format(width=patch_w, height=patch_h)

        raw_output = infer_grounded_patch_pair(
            model=model,
            processor=processor,
            patch_a=patch_a,
            patch_b=patch_b,
            question=question,
        )

        pred_count, pred_points = parse_grounded_output(
            raw_output=raw_output,
            width=patch_w,
            height=patch_h,
        )
        gt_boxes_local = patch_gt_boxes[patch_id_a]

        matches, tp, fp, fn = greedy_center_hit_match(
            pred_points=pred_points,
            gt_boxes=gt_boxes_local,
            margin_px=CENTER_HIT_MARGIN_PX,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        if pred_count is not None:
            total_pred_count += pred_count

        print(f"[patch {patch_id_a}] box_a={box_a}, box_b={box_b}")
        print(f"  pred_count(parsed)={pred_count}")
        print(f"  gt_boxes={len(gt_boxes_local)}, pred_points={len(pred_points)}")
        print(f"  tp/fp/fn={tp}/{fp}/{fn}")
        print("  raw_output:")
        if raw_output:
            for line in raw_output.splitlines():
                print(f"    {line}")
        else:
            print("    (empty)")
        if matches:
            best_distance = min(row[2] for row in matches)
            print(f"  best_center_dist_px={best_distance:.3f}")

        patch_results.append(
            {
                "patch_id": patch_id_a,
                "patch_a": patch_a,
                "patch_b": patch_b,
                "raw_output": raw_output,
                "pred_count": pred_count,
                "pred_points": pred_points,
                "gt_boxes": gt_boxes_local,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    print("\n===== SUMMARY =====")
    print("metric: center-hit (pred point inside GT box, greedy 1:1)")
    print(f"gt_answer(total changed buildings): {gt_answer}")
    print(f"total_pred_count(parsed sum): {total_pred_count}")
    print(f"total_tp={total_tp}, total_fp={total_fp}, total_fn={total_fn}")
    print(f"precision={precision:.4f}, recall={recall:.4f}")

    png_path, svg_path = save_patch_grounded_figure(
        patch_results=patch_results,
        save_path=vis_save_path,
        sample_index=SAMPLE_INDEX,
        gt_answer=gt_answer,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        precision=precision,
        recall=recall,
    )
    print(f"saved visualization (png): {png_path}")
    print(f"saved visualization (svg): {svg_path}")


if __name__ == "__main__":
    main()
