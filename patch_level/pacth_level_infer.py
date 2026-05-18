from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    logging as transformers_logging,
)


transformers_logging.set_verbosity_error()

REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKETS = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, None),
]


@dataclass(frozen=True)
class TestSample:
    image_a: str
    image_b: str
    answer: str


def suppress_noisy_logs() -> None:
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def bucket_name(value: int) -> str:
    for name, low, high in BUCKETS:
        if value >= low and (high is None or value <= high):
            return name
    raise ValueError(f"Cannot bucket value: {value}")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_rows(rows: list[dict], key: str = "index") -> dict[int, dict]:
    return {int(row[key]): row for row in rows}


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


def load_test_building_count(annotation_path: Path, data_root: Path) -> list[TestSample]:
    with annotation_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

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


def parse_first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    if match is None:
        return None
    return int(match.group(0))


def split_into_grid(
    image: Image.Image,
    patch_grid: int,
) -> list[tuple[int, tuple[int, int, int, int], Image.Image]]:
    if patch_grid < 1:
        raise ValueError(f"patch.grid must be >= 1, got {patch_grid}")

    width, height = image.size
    x_edges = [int(width * index / patch_grid) for index in range(patch_grid + 1)]
    y_edges = [int(height * index / patch_grid) for index in range(patch_grid + 1)]

    patches: list[tuple[int, tuple[int, int, int, int], Image.Image]] = []
    patch_id = 0
    for row in range(patch_grid):
        for col in range(patch_grid):
            left = x_edges[col]
            top = y_edges[row]
            right = x_edges[col + 1]
            bottom = y_edges[row + 1]
            box = (left, top, right, bottom)
            patches.append((patch_id, box, image.crop(box)))
            patch_id += 1
    return patches


def build_single_image_prompt(processor, image: Image.Image, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def move_to_model_device(inputs: dict, model) -> dict:
    return {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def first_token_id(token_id) -> int | None:
    if isinstance(token_id, (list, tuple)):
        return token_id[0] if token_id else None
    return token_id


def configure_generation_tokens(model, processor) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is None:
        pad_token_id = first_token_id(eos_token_id)

    if pad_token_id is None:
        pad_token_id = first_token_id(
            getattr(model.generation_config, "eos_token_id", None)
        )

    if pad_token_id is not None:
        model.generation_config.pad_token_id = pad_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = pad_token_id

    return pad_token_id


def infer_single_image_batch(
    model,
    processor,
    images: list[Image.Image],
    question: str,
    max_new_tokens: int,
) -> list[tuple[str, int | None]]:
    if not images:
        return []

    texts = [build_single_image_prompt(processor, image, question) for image in images]
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = move_to_model_device(inputs, model)
    input_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model.generation_config.pad_token_id,
        )

    if input_length and generated_ids.shape[-1] > input_length:
        generated_ids = generated_ids[:, input_length:]

    outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
    if len(outputs) != len(images):
        raise RuntimeError(
            f"Batch output size mismatch: got {len(outputs)}, expected {len(images)}"
        )

    return [(output.strip(), parse_first_int(output)) for output in outputs]


def summarize_patch_outputs(patch_outputs: list[dict]) -> str:
    chunks = []
    for row in patch_outputs:
        chunks.append(
            f"p{row['patch_id']}:A={row['pred_a']},B={row['pred_b']},D={row['diff']}"
        )
    return " | ".join(chunks)


def infer_patch_pair_delta(
    model,
    processor,
    sample: TestSample,
    question: str,
    patch_grid: int,
    max_new_tokens: int,
    strict_valid_patches: bool,
) -> dict:
    with Image.open(sample.image_a) as image_a_source:
        image_a = image_a_source.convert("RGB")
    with Image.open(sample.image_b) as image_b_source:
        image_b = image_b_source.convert("RGB")

    patches_a = split_into_grid(image_a, patch_grid)
    patches_b = split_into_grid(image_b, patch_grid)

    if len(patches_a) != len(patches_b):
        raise RuntimeError("Patch count mismatch between image A and B")

    results_a = infer_single_image_batch(
        model=model,
        processor=processor,
        images=[patch for _, _, patch in patches_a],
        question=question,
        max_new_tokens=max_new_tokens,
    )
    results_b = infer_single_image_batch(
        model=model,
        processor=processor,
        images=[patch for _, _, patch in patches_b],
        question=question,
        max_new_tokens=max_new_tokens,
    )

    total_patches = len(patches_a)
    valid_patch_pairs = 0
    sum_count_a = 0
    sum_count_b = 0
    sum_abs_diff = 0
    sum_diff_valid = 0
    patch_outputs: list[dict] = []

    for (patch_id_a, box_a, _), (patch_id_b, box_b, _), (raw_a, pred_a), (
        raw_b,
        pred_b,
    ) in zip(patches_a, patches_b, results_a, results_b):
        if patch_id_a != patch_id_b:
            raise RuntimeError("Patch id mismatch between A and B")

        diff = pred_b - pred_a if pred_a is not None and pred_b is not None else None

        if pred_a is not None:
            sum_count_a += pred_a
        if pred_b is not None:
            sum_count_b += pred_b
        if diff is not None:
            valid_patch_pairs += 1
            sum_abs_diff += abs(diff)
            sum_diff_valid += diff

        patch_outputs.append(
            {
                "patch_id": patch_id_a,
                "box_a": box_a,
                "box_b": box_b,
                "raw_a": raw_a,
                "raw_b": raw_b,
                "pred_a": pred_a,
                "pred_b": pred_b,
                "diff": diff,
            }
        )

    if strict_valid_patches and valid_patch_pairs != total_patches:
        pred = None
    elif valid_patch_pairs == 0:
        pred = None
    else:
        pred = sum_diff_valid

    return {
        "pred": pred,
        "total_patches": total_patches,
        "valid_patch_pairs": valid_patch_pairs,
        "sum_count_a": sum_count_a,
        "sum_count_b": sum_count_b,
        "sum_abs_diff": sum_abs_diff,
        "sum_diff_valid": sum_diff_valid,
        "patch_outputs": patch_outputs,
        "raw_output": summarize_patch_outputs(patch_outputs),
    }


def batched(items: list[tuple[int, TestSample]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@hydra.main(version_base=None, config_path="configs", config_name="patch_level")
def main(cfg: DictConfig) -> None:
    suppress_noisy_logs()

    data_root = repo_path(cfg.data.root)
    annotation_path = repo_path(cfg.data.annotation_path)
    output_dir = repo_path(cfg.output.root) / cfg.model.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / cfg.output.predictions_jsonl

    samples = load_test_building_count(
        annotation_path=annotation_path,
        data_root=data_root,
    )
    if cfg.inference.limit is not None:
        samples = samples[: cfg.inference.limit]

    done = index_rows(load_jsonl(jsonl_path)) if cfg.inference.resume else {}

    print(f"model: {cfg.model.title}")
    print(f"model id: {cfg.model.model_id}")
    print(f"output dir: {output_dir}")
    print(f"test samples: {len(samples)}")
    print(f"already predicted: {len(done)}")
    print(f"patch grid: {cfg.patch.grid}x{cfg.patch.grid}")
    print(f"strict_valid_patches: {cfg.patch.strict_valid_patches}")

    if not cfg.inference.resume and jsonl_path.exists():
        jsonl_path.unlink()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(cfg.model.model_id, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        tie_word_embeddings=False,
        trust_remote_code=True,
    ).eval()
    configure_generation_tokens(model, processor)

    pending = [
        (index, sample)
        for index, sample in enumerate(samples)
        if index not in done
    ]

    started = time.time()
    progress = tqdm(
        total=len(pending),
        desc="patch-level building-count test",
        unit="sample",
    )

    for batch in batched(pending, cfg.inference.batch_size):
        for index, sample in batch:
            sample_started = time.time()
            result = infer_patch_pair_delta(
                model=model,
                processor=processor,
                sample=sample,
                question=cfg.question,
                patch_grid=int(cfg.patch.grid),
                max_new_tokens=int(cfg.inference.max_new_tokens),
                strict_valid_patches=bool(cfg.patch.strict_valid_patches),
            )
            elapsed_per_sample = time.time() - sample_started

            gt = parse_first_int(sample.answer)
            if gt is None:
                raise ValueError(
                    f"GT answer is not integer-parsable at index={index}: {sample.answer}"
                )

            row = {
                "index": index,
                "model": cfg.model.key,
                "model_id": cfg.model.model_id,
                "image_a": sample.image_a,
                "image_b": sample.image_b,
                "gt": gt,
                "pred": result["pred"],
                "correct": result["pred"] == gt,
                "bucket": bucket_name(gt),
                "raw_output": result["raw_output"],
                "elapsed_sec": round(elapsed_per_sample, 4),
                "patch_grid": int(cfg.patch.grid),
                "total_patches": result["total_patches"],
                "valid_patch_pairs": result["valid_patch_pairs"],
                "sum_count_a": result["sum_count_a"],
                "sum_count_b": result["sum_count_b"],
                "sum_abs_diff": result["sum_abs_diff"],
                "sum_diff_valid": result["sum_diff_valid"],
                "patch_outputs": result["patch_outputs"],
            }
            append_jsonl(jsonl_path, row)
            done[index] = row

            total_elapsed = time.time() - started
            progress.update(1)
            progress.set_postfix(
                done=f"{len(done)}/{len(samples)}",
                elapsed=f"{total_elapsed / 60:.1f}m",
            )

    progress.close()

    print(f"saved: {jsonl_path}")


if __name__ == "__main__":
    main()
