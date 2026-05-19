from __future__ import annotations

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
