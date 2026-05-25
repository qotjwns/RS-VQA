from __future__ import annotations

from .batching import batched
from .change_mask import Component, connected_components, load_binary_mask, resolve_mask_paths
from .constants import BUCKETS, PREDICTION_FIELDNAMES
from .data import TestSample, load_test_building_count
from .io import append_jsonl, index_rows, load_jsonl
from .logging_utils import suppress_http_logs
from .model import configure_generation_tokens, first_token_id, move_to_model_device
from .paths import REPO_ROOT, repo_path, resolve_image_path
from .prompt import build_prompt_with_images
from .text import parse_first_int

__all__ = [
    "REPO_ROOT",
    "BUCKETS",
    "Component",
    "PREDICTION_FIELDNAMES",
    "TestSample",
    "append_jsonl",
    "batched",
    "build_prompt_with_images",
    "bucket_name",
    "configure_generation_tokens",
    "connected_components",
    "first_token_id",
    "index_rows",
    "load_jsonl",
    "load_binary_mask",
    "load_test_building_count",
    "move_to_model_device",
    "parse_first_int",
    "repo_path",
    "resolve_image_path",
    "resolve_mask_paths",
    "suppress_http_logs",
]


def bucket_name(value: int) -> str:
    for name, low, high in BUCKETS:
        if value >= low and (high is None or value <= high):
            return name
    raise ValueError(f"Cannot bucket value: {value}")
