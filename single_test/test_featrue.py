from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

matplotlib.use("Agg")

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


# Change only these constants for another image pair.
IMAGE_T1_PATH = repo_path("data/coding/datasets/LEVIR-MCI-dataset/images/test/A/test_000300.png")
IMAGE_T2_PATH = repo_path("data/coding/datasets/LEVIR-MCI-dataset/images/test/B/test_000300.png")
MODEL_ID = "nvidia/segformer-b1-finetuned-ade-512-512"
OUTPUT_DIR = repo_path("outputs/segformer_level_test")
ANNOTATION_PATH = repo_path("data/coding/muti_task_data/test_task_data/count_build.json")

PT_SAVE_PATH = OUTPUT_DIR / "test_featrue_segformer_level_features.pt"
JSON_SAVE_PATH = OUTPUT_DIR / "test_featrue_segformer_level_features.json"
VIS_SAVE_PATH = OUTPUT_DIR / "test_featrue_segformer_level_visualization.png"

DEVICE_NAME = "auto"
DTYPE_NAME = "auto"


def select_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def load_segformer_encoder(model_id: str, dtype: torch.dtype) -> torch.nn.Module:
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(model_id, dtype=dtype)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = SegformerForSemanticSegmentation.from_pretrained(model_id, torch_dtype=dtype)
    return model.segformer.eval()


def move_inputs_to_device(
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=dtype)
        elif torch.is_tensor(value):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def extract_level_feature_maps(
    encoder: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], dict[str, object]]:
    pixel_values = inputs["pixel_values"]

    with torch.inference_mode():
        outputs = encoder(**inputs, output_hidden_states=True)

    level_feature_maps = list(outputs.hidden_states)
    if len(level_feature_maps) != 4:
        raise RuntimeError(f"Expected 4 SegFormer levels, got {len(level_feature_maps)}")

    metadata = {
        "model_id": MODEL_ID,
        "image_t1_path": str(IMAGE_T1_PATH),
        "image_t2_path": str(IMAGE_T2_PATH),
        "pixel_values_shape": list(pixel_values.shape),
        "level_feature_map_shapes": [list(feature.shape) for feature in level_feature_maps],
        "level_order": ["level_1", "level_2", "level_3", "level_4"],
        "level_note": (
            "SegFormer returns four hierarchical encoder feature maps. "
            "Level 1 has the highest spatial resolution and more low-level detail. "
            "Level 4 has the lowest spatial resolution and stronger semantic abstraction."
        ),
    }
    return level_feature_maps, metadata


def load_gt_count(annotation_path: Path, image_t1_path: Path, image_t2_path: Path) -> int | None:
    if not annotation_path.exists():
        return None

    records = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_t1_name = image_t1_path.name
    image_t2_name = image_t2_path.name
    for record in records:
        image_paths = record.get("images", [])
        if len(image_paths) != 2:
            continue
        if Path(image_paths[0]).name != image_t1_name or Path(image_paths[1]).name != image_t2_name:
            continue

        conversations = record.get("conversations", [])
        if not conversations:
            return None
        answer = conversations[-1].get("value", "").strip()
        try:
            return int(answer)
        except ValueError:
            return None

    return None


def resolve_gt_mask_paths(image_t1_path: Path) -> tuple[Path | None, Path | None]:
    test_dir = image_t1_path.parent.parent
    label_path = test_dir / "label" / image_t1_path.name
    label_rgb_path = test_dir / "label_rgb" / image_t1_path.name
    return (
        label_path if label_path.exists() else None,
        label_rgb_path if label_rgb_path.exists() else None,
    )


def load_gt_mask_display(label_path: Path | None, label_rgb_path: Path | None) -> Image.Image | None:
    mask_path = label_rgb_path or label_path
    if mask_path is None:
        return None
    return Image.open(mask_path).convert("RGB")


def feature_map_to_pca_rgb(feature_map: torch.Tensor) -> torch.Tensor:
    """Convert one (C, H, W) feature map to a 3-channel PCA preview."""
    channels, height, width = feature_map.shape
    features = feature_map.permute(1, 2, 0).reshape(height * width, channels).float()
    features = features - features.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(features, q=3, center=False)
    preview = features @ components[:, :3]
    preview = preview.reshape(height, width, 3)

    min_value = preview.amin(dim=(0, 1), keepdim=True)
    max_value = preview.amax(dim=(0, 1), keepdim=True)
    preview = (preview - min_value) / (max_value - min_value).clamp_min(1e-6)
    return preview.clamp(0, 1)


def cosine_similarity_map(feature_map_t1: torch.Tensor, feature_map_t2: torch.Tensor) -> torch.Tensor:
    """Pixel-wise cosine similarity for two same-level (C, H, W) feature maps."""
    t1 = F.normalize(feature_map_t1.float(), dim=0)
    t2 = F.normalize(feature_map_t2.float(), dim=0)
    return (t1 * t2).sum(dim=0)


def make_cosine_maps(level_feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
    return [
        cosine_similarity_map(feature_map[0], feature_map[1]).detach().cpu()
        for feature_map in level_feature_maps
    ]


def global_cosine_similarity_by_level(level_feature_maps: list[torch.Tensor]) -> torch.Tensor:
    similarities = []
    for feature_map in level_feature_maps:
        pooled = feature_map.mean(dim=(-2, -1))
        pooled = F.normalize(pooled.float(), dim=-1)
        similarities.append((pooled[0] * pooled[1]).sum(dim=-1))
    return torch.stack(similarities).detach().cpu()


def save_level_visualization(
    images: list[Image.Image],
    level_feature_maps: list[torch.Tensor],
    cosine_maps: list[torch.Tensor],
    cosine_by_level: torch.Tensor,
    gt_count: int | None,
    gt_mask: Image.Image | None,
    save_path: Path,
) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_levels = len(level_feature_maps)
    rows = num_levels + 1
    fig, axes = plt.subplots(rows, 4, figsize=(19, 4 + 3.3 * num_levels))

    axes[0, 0].imshow(images[0])
    axes[0, 0].set_title("T1 image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(images[1])
    axes[0, 1].set_title("T2 image")
    axes[0, 1].axis("off")

    axes[0, 2].axis("off")
    if gt_mask is not None:
        axes[0, 2].imshow(gt_mask)
        axes[0, 2].set_title(f"GT change mask | count={gt_count}")
        axes[0, 2].axis("off")
    else:
        axes[0, 2].text(0.0, 0.5, f"GT mask not found\nGT count={gt_count}", fontsize=11)

    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.0,
        0.95,
        (
            "{ f_1^(1), f_2^(1), f_3^(1), f_4^(1) } = Ftheta(I_t1)\n"
            "{ f_1^(2), f_2^(2), f_3^(2), f_4^(2) } = Ftheta(I_t2)\n\n"
            "Ftheta = SegFormer-B1 encoder\n"
            "Feature maps = PCA previews of encoder levels\n"
            "Cosine map = same-level T1/T2 feature similarity\n\n"
            "Bright cosine = similar, dark cosine = different\n"
            f"GT changed buildings = {gt_count}"
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    for level_index, feature_map in enumerate(level_feature_maps):
        row = level_index + 1
        feature_t1 = feature_map[0]
        feature_t2 = feature_map[1]
        preview_t1 = feature_map_to_pca_rgb(feature_t1)
        preview_t2 = feature_map_to_pca_rgb(feature_t2)
        cosine_map = cosine_maps[level_index]
        mean_cosine = cosine_by_level[level_index].item()
        _, height, width = feature_t1.shape

        axes[row, 0].imshow(preview_t1.numpy(), interpolation="nearest")
        axes[row, 0].set_title(f"T1 f_{level_index + 1} | {tuple(feature_t1.shape)}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(preview_t2.numpy(), interpolation="nearest")
        axes[row, 1].set_title(f"T2 f_{level_index + 1} | {tuple(feature_t2.shape)}")
        axes[row, 1].axis("off")

        image = axes[row, 2].imshow(
            cosine_map.numpy(),
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 2].set_title(
            f"cosine level {level_index + 1} | {height}x{width} | mean={mean_cosine:.4f}"
        )
        axes[row, 2].axis("off")
        fig.colorbar(image, ax=axes[row, 2], fraction=0.046, pad=0.04)

        if gt_mask is not None:
            gt_level = gt_mask.resize((width, height), Image.Resampling.NEAREST)
            axes[row, 3].imshow(gt_level, interpolation="nearest")
            axes[row, 3].set_title(f"GT mask resized to level {level_index + 1} | {height}x{width}")
        else:
            axes[row, 3].text(0.0, 0.5, "GT mask not found", fontsize=11)
        axes[row, 3].axis("off")

    fig.suptitle("SegFormer Level Feature Maps and T1/T2 Cosine Similarity", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def print_summary(level_feature_maps: list[torch.Tensor], cosine_by_level: torch.Tensor) -> None:
    print()
    print("Formula")
    print("{ f_1^(1), f_2^(1), f_3^(1), f_4^(1) } = Ftheta(I_t1)")
    print("{ f_1^(2), f_2^(2), f_3^(2), f_4^(2) } = Ftheta(I_t2)")
    print()
    print("Meaning")
    print("Ftheta  = SegFormer-B1 encoder")
    print("I_t1    = T1 image")
    print("I_t2    = T2 image")
    print("f_l^(1) = T1 image level-l feature map")
    print("f_l^(2) = T2 image level-l feature map")
    print()
    print("Level shapes")
    for level_index, feature_map in enumerate(level_feature_maps, start=1):
        print(f"level {level_index}: {tuple(feature_map.shape)}")
    print()
    print("Global cosine similarity by level")
    for level_index, value in enumerate(cosine_by_level.tolist(), start=1):
        print(f"level {level_index}: {value:.4f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = select_device(DEVICE_NAME)
    dtype = select_dtype(DTYPE_NAME, device)
    images = [
        Image.open(IMAGE_T1_PATH).convert("RGB"),
        Image.open(IMAGE_T2_PATH).convert("RGB"),
    ]
    gt_count = load_gt_count(ANNOTATION_PATH, IMAGE_T1_PATH, IMAGE_T2_PATH)
    label_path, label_rgb_path = resolve_gt_mask_paths(IMAGE_T1_PATH)
    gt_mask = load_gt_mask_display(label_path, label_rgb_path)

    print("Image t1:", IMAGE_T1_PATH)
    print("Image t2:", IMAGE_T2_PATH)
    print("GT count:", gt_count)
    print("GT label:", label_path)
    print("GT label_rgb:", label_rgb_path)
    print("Original sizes:", [image.size for image in images])
    print("Model:", MODEL_ID)
    print("Device:", device)
    print("Dtype:", dtype)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    encoder = load_segformer_encoder(MODEL_ID, dtype=dtype).to(device)
    inputs = processor(images=images, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, device=device, dtype=dtype)

    level_feature_maps, metadata = extract_level_feature_maps(encoder=encoder, inputs=inputs)
    level_feature_maps_cpu = [feature.detach().cpu() for feature in level_feature_maps]
    cosine_maps = make_cosine_maps(level_feature_maps)
    cosine_by_level = global_cosine_similarity_by_level(level_feature_maps)

    vis_path = save_level_visualization(
        images=images,
        level_feature_maps=level_feature_maps_cpu,
        cosine_maps=cosine_maps,
        cosine_by_level=cosine_by_level,
        gt_count=gt_count,
        gt_mask=gt_mask,
        save_path=VIS_SAVE_PATH,
    )

    metadata = {
        **metadata,
        "original_image_sizes": [list(image.size) for image in images],
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "annotation_path": str(ANNOTATION_PATH),
        "gt_changed_building_count": gt_count,
        "gt_label_path": str(label_path) if label_path is not None else None,
        "gt_label_rgb_path": str(label_rgb_path) if label_rgb_path is not None else None,
        "global_cosine_similarity_by_level": cosine_by_level.tolist(),
        "cosine_map_shapes": [list(cosine_map.shape) for cosine_map in cosine_maps],
        "visualization_path": str(vis_path),
    }
    torch.save(
        {
            "level_feature_maps": level_feature_maps_cpu,
            "features_t1": [feature[0] for feature in level_feature_maps_cpu],
            "features_t2": [feature[1] for feature in level_feature_maps_cpu],
            "cosine_maps": cosine_maps,
            "global_cosine_similarity_by_level": cosine_by_level,
            "metadata": metadata,
        },
        PT_SAVE_PATH,
    )
    JSON_SAVE_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print_summary(level_feature_maps_cpu, cosine_by_level)
    print()
    print("Saved")
    print("PT:", PT_SAVE_PATH)
    print("metadata:", JSON_SAVE_PATH)
    print("visualization:", vis_path)


if __name__ == "__main__":
    main()
