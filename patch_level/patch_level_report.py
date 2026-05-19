from __future__ import annotations

import csv
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util import BUCKETS, load_jsonl, repo_path


PREFERRED_FIELDNAMES = [
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
    "patch_grid",
    "total_patches",
    "valid_patch_pairs",
    "sum_count_a",
    "sum_count_b",
    "sum_diff_valid",
    "sum_abs_diff",
    "patch_outputs",
]


def load_predictions(path: Path) -> list[dict]:
    rows = load_jsonl(path)
    return sorted(rows, key=lambda row: int(row["index"]))


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def ordered_fieldnames(rows: list[dict]) -> list[str]:
    discovered: dict[str, None] = {}
    for row in rows:
        for key in row.keys():
            discovered.setdefault(key, None)

    remaining = [key for key in discovered.keys() if key not in PREFERRED_FIELDNAMES]
    return [
        key for key in PREFERRED_FIELDNAMES if key in discovered
    ] + sorted(remaining)


def stringify_cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ordered_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify_cell(value) for key, value in row.items()})


def summarize_by_bucket(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for name, low, high in BUCKETS:
        bucket_rows = [
            row
            for row in rows
            if int(row["gt"]) >= low and (high is None or int(row["gt"]) <= high)
        ]

        valid_rows = [row for row in bucket_rows if is_number(row.get("pred"))]
        correct = sum(1 for row in bucket_rows if row.get("pred") == row.get("gt"))
        mae = (
            sum(abs(int(row["pred"]) - int(row["gt"])) for row in valid_rows)
            / len(valid_rows)
            if valid_rows
            else 0.0
        )
        accuracy = correct / len(bucket_rows) if bucket_rows else 0.0

        patch_ratio_values = [
            row["valid_patch_pairs"] / row["total_patches"]
            for row in bucket_rows
            if is_number(row.get("valid_patch_pairs"))
            and is_number(row.get("total_patches"))
            and row["total_patches"] > 0
        ]
        avg_valid_patch_ratio = (
            sum(patch_ratio_values) / len(patch_ratio_values)
            if patch_ratio_values
            else 0.0
        )

        patch_abs_diff_values = [
            float(row["sum_abs_diff"])
            for row in bucket_rows
            if is_number(row.get("sum_abs_diff"))
        ]
        avg_sum_abs_diff = (
            sum(patch_abs_diff_values) / len(patch_abs_diff_values)
            if patch_abs_diff_values
            else 0.0
        )

        summary.append(
            {
                "bucket": name,
                "n": len(bucket_rows),
                "strict_accuracy": accuracy,
                "mae": mae,
                "invalid": len(bucket_rows) - len(valid_rows),
                "avg_valid_patch_ratio": avg_valid_patch_ratio,
                "avg_sum_abs_diff": avg_sum_abs_diff,
            }
        )

    return summary


def save_summary_csv(path: Path, summary: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "n",
                "strict_accuracy",
                "mae",
                "invalid",
                "avg_valid_patch_ratio",
                "avg_sum_abs_diff",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def ensure_matplotlib_cache_dir() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return

    default_dir = Path.home() / ".matplotlib"
    if default_dir.exists() and os.access(default_dir, os.W_OK):
        return

    cache_dir = Path(tempfile.gettempdir()) / "rs_vqa_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def suppress_noisy_logs() -> None:
    for logger_name in ("matplotlib", "matplotlib.font_manager"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def draw_bucket_plot(path: Path, summary: list[dict], title: str) -> None:
    ensure_matplotlib_cache_dir()
    import matplotlib.pyplot as plt

    labels = [row["bucket"] for row in summary]
    counts = [row["n"] for row in summary]
    accuracies = [row["strict_accuracy"] * 100 for row in summary]
    maes = [row["mae"] for row in summary]
    valid_ratios = [row["avg_valid_patch_ratio"] * 100 for row in summary]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle(f"{title} - patch-level per-bucket performance", fontsize=14)

    axes[0].bar(labels, accuracies, color="#4285F4", edgecolor="black")
    axes[0].set_title("Strict accuracy by GT bucket")
    axes[0].set_ylabel("Strict accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25)
    for index, (accuracy, count) in enumerate(zip(accuracies, counts)):
        axes[0].text(
            index,
            min(accuracy + 2, 97),
            f"{accuracy:.0f}\\n(n={count})",
            ha="center",
            va="bottom",
        )

    axes[1].bar(labels, maes, color="#F94144", edgecolor="black")
    axes[1].set_title("MAE by GT bucket")
    axes[1].set_ylabel("MAE (|pred - gt|)")
    axes[1].grid(axis="y", alpha=0.25)
    max_mae = max(maes) if maes else 0
    axes[1].set_ylim(0, max_mae * 1.18 if max_mae > 0 else 1)
    for index, mae in enumerate(maes):
        axes[1].text(
            index,
            mae + max(max_mae * 0.03, 0.05),
            f"{mae:.2f}",
            ha="center",
            va="bottom",
        )

    axes[2].bar(labels, valid_ratios, color="#43AA8B", edgecolor="black")
    axes[2].set_title("Average valid patch ratio")
    axes[2].set_ylabel("Valid patch ratio (%)")
    axes[2].set_ylim(0, 100)
    axes[2].grid(axis="y", alpha=0.25)
    for index, ratio in enumerate(valid_ratios):
        axes[2].text(
            index,
            min(ratio + 2, 97),
            f"{ratio:.0f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=200)
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="patch_level")
def main(cfg: DictConfig) -> None:
    suppress_noisy_logs()

    output_dir = repo_path(cfg.output.root) / cfg.model.output_name
    predictions_jsonl = output_dir / cfg.output.predictions_jsonl
    predictions_csv = output_dir / cfg.output.predictions_csv
    summary_csv = output_dir / cfg.output.summary_csv
    plot_path = output_dir / cfg.output.plot_path

    if not predictions_jsonl.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_jsonl}")

    rows = load_predictions(predictions_jsonl)
    if not rows:
        raise ValueError(f"No rows found in prediction file: {predictions_jsonl}")

    save_predictions_csv(predictions_csv, rows)

    summary = summarize_by_bucket(rows)
    save_summary_csv(summary_csv, summary)
    draw_bucket_plot(plot_path, summary, cfg.model.title)

    print(f"model: {cfg.model.title}")
    print(f"loaded: {predictions_jsonl}")
    print(f"saved: {predictions_csv}")
    print(f"saved: {summary_csv}")
    print(f"saved: {plot_path}")


if __name__ == "__main__":
    main()
