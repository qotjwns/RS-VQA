from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Component:
    area: int
    centroid_y: float
    centroid_x: float


def resolve_mask_paths(image_a_path: Path) -> tuple[Path | None, Path | None]:
    parts = list(image_a_path.parts)
    candidates = [idx for idx, token in enumerate(parts) if token in {"A", "B"}]
    for idx in reversed(candidates):
        label_path = Path(*parts[:idx], "label", *parts[idx + 1 :])
        label_rgb_path = Path(*parts[:idx], "label_rgb", *parts[idx + 1 :])
        return (
            label_path if label_path.exists() else None,
            label_rgb_path if label_rgb_path.exists() else None,
        )
    raise FileNotFoundError(f"Cannot infer mask directories from: {image_a_path}")


def load_binary_mask(
    label_path: Path | None,
    label_rgb_path: Path | None,
    rgb_red_min: int = 160,
    rgb_green_max: int = 120,
    rgb_blue_max: int = 120,
    red_dominance_margin: int = 40,
) -> tuple[np.ndarray, str]:
    gray_mask: np.ndarray | None = None
    red_mask: np.ndarray | None = None
    rgb_nonzero_mask: np.ndarray | None = None

    if label_path is not None:
        with Image.open(label_path) as mask_file:
            gray = np.array(mask_file)
        if gray.ndim == 2:
            gray_mask = gray > 0
        elif gray.ndim == 3:
            gray_mask = np.any(gray > 0, axis=-1)
        else:
            raise ValueError(f"Unsupported gray label shape: {gray.shape} ({label_path})")

    if label_rgb_path is not None:
        with Image.open(label_rgb_path) as rgb_file:
            rgb = np.array(rgb_file)
        if rgb.ndim == 2:
            rgb_nonzero_mask = rgb > 0
        elif rgb.ndim == 3:
            r = rgb[..., 0].astype(np.int16)
            g = rgb[..., 1].astype(np.int16)
            b = rgb[..., 2].astype(np.int16)
            rgb_nonzero_mask = np.any(rgb > 0, axis=-1)
            red_mask = (
                (r >= rgb_red_min)
                & (g <= rgb_green_max)
                & (b <= rgb_blue_max)
                & ((r - g) >= red_dominance_margin)
                & ((r - b) >= red_dominance_margin)
            )
        else:
            raise ValueError(f"Unsupported rgb label shape: {rgb.shape} ({label_rgb_path})")

    if gray_mask is not None and red_mask is not None:
        if gray_mask.shape != red_mask.shape:
            raise ValueError(
                f"label/label_rgb shape mismatch: {gray_mask.shape} vs {red_mask.shape}"
            )
        intersection = gray_mask & red_mask
        if np.any(intersection):
            return intersection.astype(np.uint8), "label & label_rgb(red)"
        if np.any(red_mask):
            return red_mask.astype(np.uint8), "label_rgb(red)"
        return gray_mask.astype(np.uint8), "label(gray)"

    if red_mask is not None and np.any(red_mask):
        return red_mask.astype(np.uint8), "label_rgb(red)"

    if gray_mask is not None:
        return gray_mask.astype(np.uint8), "label(gray)"

    if rgb_nonzero_mask is not None:
        return rgb_nonzero_mask.astype(np.uint8), "label_rgb(nonzero-fallback)"

    raise FileNotFoundError("No usable mask found (both label and label_rgb are missing).")


def connected_components(binary: np.ndarray, connectivity: int = 8) -> list[Component]:
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
    components: list[Component] = []

    for y, x in zip(ys.tolist(), xs.tolist()):
        if visited[y, x]:
            continue

        stack = [(y, x)]
        visited[y, x] = True
        pixels_y: list[int] = []
        pixels_x: list[int] = []

        while stack:
            cy, cx = stack.pop()
            pixels_y.append(cy)
            pixels_x.append(cx)
            for dy, dx in offsets:
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    continue
                if visited[ny, nx] or not binary[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        area = len(pixels_y)
        components.append(
            Component(
                area=area,
                centroid_y=float(sum(pixels_y)) / area,
                centroid_x=float(sum(pixels_x)) / area,
            )
        )

    return components
