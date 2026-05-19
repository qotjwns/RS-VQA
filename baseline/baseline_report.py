from __future__ import annotations

import csv
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util import (
    BUCKETS,
    PREDICTION_FIELDNAMES,
    load_jsonl,
    repo_path,
)

# JSONL 예측 결과를 불러온 뒤 index 기준으로 정렬
def load_predictions(path: Path) -> list[dict]:
    rows = load_jsonl(path)
    return sorted(rows, key=lambda row: int(row["index"]))

# 예측 결과 리스트를 CSV 파일로 저장
def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=PREDICTION_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

# GT 값을 bucket 범위별로 나누어 샘플 수, 정확도, MAE, invalid 개수를 계산
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

# bucket별 성능 요약 결과를 CSV 파일로 저장
def save_summary_csv(path: Path, summary: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bucket", "n", "strict_accuracy", "mae", "invalid"],
        )
        writer.writeheader()
        writer.writerows(summary)

# bucket별 strict accuracy와 MAE를 막대그래프로 시각화하여 이미지 파일로 저장
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

# Hydra 설정을 불러와 예측 결과를 CSV, summary CSV, 성능 그래프로 변환하는 전체 평가 후처리 과정을 실행
@hydra.main(version_base=None, config_path="configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    output_dir = repo_path(cfg.output.root) / cfg.model.output_name
    predictions_jsonl = output_dir / cfg.output.predictions_jsonl
    predictions_csv = output_dir / cfg.output.predictions_csv
    summary_csv = output_dir / cfg.output.summary_csv
    plot_path = output_dir / cfg.output.plot_path

    if not predictions_jsonl.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_jsonl}")

    rows = load_predictions(predictions_jsonl)
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
