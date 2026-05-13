from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "building_count_test"
PREDICTIONS_JSONL = OUTPUT_DIR / "test_predictions.jsonl"
PREDICTIONS_CSV = OUTPUT_DIR / "test_predictions.csv"
SUMMARY_CSV = OUTPUT_DIR / "test_bucket_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "test_bucket_summary.md"
PLOT_PATH = OUTPUT_DIR / "test_bucket_performance.png"
TITLE = "internVL3.5-8B"
BUCKETS = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, None),
]


def load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return sorted(rows, key=lambda row: int(row["index"]))


def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "index",
        "image_a",
        "image_b",
        "gt",
        "pred",
        "correct",
        "bucket",
        "raw_output",
        "elapsed_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_bucket(rows: list[dict]) -> list[dict]:
    summary = []
    for name, low, high in BUCKETS:
        bucket_rows = [
            row
            for row in rows
            if int(row["gt"]) >= low and (high is None or int(row["gt"]) <= high)
        ]
        valid_rows = [row for row in bucket_rows if row["pred"] is not None]
        correct = sum(1 for row in bucket_rows if row["pred"] == row["gt"])
        mae = (
            sum(abs(int(row["pred"]) - int(row["gt"])) for row in valid_rows)
            / len(valid_rows)
            if valid_rows
            else 0.0
        )
        accuracy = correct / len(bucket_rows) if bucket_rows else 0.0
        summary.append(
            {
                "bucket": name,
                "n": len(bucket_rows),
                "strict_accuracy": accuracy,
                "mae": mae,
                "invalid": len(bucket_rows) - len(valid_rows),
            }
        )
    return summary


def save_summary_csv(path: Path, summary: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bucket", "n", "strict_accuracy", "mae", "invalid"],
        )
        writer.writeheader()
        writer.writerows(summary)


def save_summary_markdown(path: Path, summary: list[dict]) -> None:
    lines = [
        "| GT bucket | n | Strict accuracy | MAE | Invalid |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {bucket} | {n} | {acc:.2f}% | {mae:.2f} | {invalid} |".format(
                bucket=row["bucket"],
                n=row["n"],
                acc=row["strict_accuracy"] * 100,
                mae=row["mae"],
                invalid=row["invalid"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_bucket_plot(path: Path, summary: list[dict], title: str) -> None:
    import matplotlib.pyplot as plt

    labels = [row["bucket"] for row in summary]
    counts = [row["n"] for row in summary]
    accuracies = [row["strict_accuracy"] * 100 for row in summary]
    maes = [row["mae"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{title} - per-bucket performance", fontsize=14)

    axes[0].bar(labels, accuracies, color="#4285F4", edgecolor="black")
    axes[0].set_title("Strict accuracy by GT bucket")
    axes[0].set_ylabel("Strict accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25)
    for index, (accuracy, count) in enumerate(zip(accuracies, counts)):
        axes[0].text(
            index,
            min(accuracy + 2, 97),
            f"{accuracy:.0f}%\n(n={count})",
            ha="center",
            va="bottom",
        )

    axes[1].bar(labels, maes, color="#F94144", edgecolor="black")
    axes[1].set_title("Mean absolute error by GT bucket")
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

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    if not PREDICTIONS_JSONL.exists():
        raise FileNotFoundError(f"Prediction file not found: {PREDICTIONS_JSONL}")

    rows = load_predictions(PREDICTIONS_JSONL)
    save_predictions_csv(PREDICTIONS_CSV, rows)

    summary = summarize_by_bucket(rows)
    save_summary_csv(SUMMARY_CSV, summary)
    save_summary_markdown(SUMMARY_MD, summary)
    draw_bucket_plot(PLOT_PATH, summary, TITLE)

    print(f"loaded: {PREDICTIONS_JSONL}")
    print(f"saved: {PREDICTIONS_CSV}")
    print(f"saved: {SUMMARY_CSV}")
    print(f"saved: {SUMMARY_MD}")
    print(f"saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
