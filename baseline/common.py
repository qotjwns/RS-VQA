from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKETS = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, None),
]

PREDICTION_FIELDNAMES = [
    "index",
    "model",
    "model_id",
    "image_a",
    "image_b",
    "gt",
    "pred",
    "correct",
    "bucket",
    "raw_output",
    "elapsed_sec",
]

# 입력된 경로가 절대경로이면 그대로 반환하고, 상대경로이면 REPO_ROOT 기준 경로로 변환
def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path

# 입력된 정수 value가 속하는 범위의 bucket 이름을 찾아 반환하고, 해당 범위가 없으면 오류를 발생
def bucket_name(value: int) -> str:
    for name, low, high in BUCKETS:
        if value >= low and (high is None or value <= high):
            return name
    raise ValueError(f"Cannot bucket value: {value}")

# JSONL 파일을 한 줄씩 읽어 dict 리스트로 반환하고, 파일이 없으면 빈 리스트를 반환
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

# 하나의 dict 데이터를 JSON 문자열로 변환하여 JSONL 파일 끝에 한 줄 추가
def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# dict 리스트를 지정한 key 값 기준으로 인덱싱하여 {index: row} 형태의 딕셔너리로 변환
def index_rows(rows: list[dict], key: str = "index") -> dict[int, dict]:
    return {int(row[key]): row for row in rows}
