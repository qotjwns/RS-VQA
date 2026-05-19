from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


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
