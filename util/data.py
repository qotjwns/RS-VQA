from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .paths import resolve_image_path


@dataclass(frozen=True)
class TestSample:
    image_a: str
    image_b: str
    answer: str


def load_test_building_count(annotation_path: Path, data_root: Path) -> list[TestSample]:
    with annotation_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    samples: list[TestSample] = []
    for record in records:
        image_a, image_b = record["images"]
        samples.append(
            TestSample(
                image_a=str(resolve_image_path(image_a, data_root)),
                image_b=str(resolve_image_path(image_b, data_root)),
                answer=str(record["conversations"][1]["value"]).strip(),
            )
        )
    return samples
