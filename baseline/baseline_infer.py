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

REPO_ROOT = Path(__file__).resolve().parents[1]
BUCKETS = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21+", 21, None),
]


transformers_logging.set_verbosity_error()


def suppress_noisy_logs() -> None:
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


@dataclass(frozen=True)
class TestSample:
    image_a: str
    image_b: str
    answer: str


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


def bucket_name(value: int) -> str:
    for name, low, high in BUCKETS:
        if value >= low and (high is None or value <= high):
            return name
    raise ValueError(f"Cannot bucket value: {value}")


def load_done(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}

    done: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            done[int(row["index"])] = row
    return done


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(
    processor,
    image_a: Image.Image,
    image_b: Image.Image,
    question: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_a},
                {"type": "image", "image": image_b},
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
        pad_token_id = first_token_id(getattr(model.generation_config, "eos_token_id", None))

    if pad_token_id is not None:
        model.generation_config.pad_token_id = pad_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = pad_token_id

    return pad_token_id


def infer_batch(
    model,
    processor,
    samples: list[TestSample],
    question: str,
    max_new_tokens: int,
) -> list[tuple[str, int | None]]:
    image_pairs = [
        (
            Image.open(sample.image_a).convert("RGB"),
            Image.open(sample.image_b).convert("RGB"),
        )
        for sample in samples
    ]
    texts = [
        build_prompt(processor, image_a, image_b, question)
        for image_a, image_b in image_pairs
    ]
    nested_images = [[image_a, image_b] for image_a, image_b in image_pairs]

    try:
        inputs = processor(
            text=texts,
            images=nested_images,
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        flat_images = [image for pair in nested_images for image in pair]
        inputs = processor(
            text=texts,
            images=flat_images,
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
    if len(outputs) != len(samples):
        raise RuntimeError(
            f"Batch output size mismatch: got {len(outputs)}, expected {len(samples)}"
        )

    return [(output.strip(), parse_first_int(output)) for output in outputs]


def batched(items: list[tuple[int, TestSample]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@hydra.main(version_base=None, config_path="configs", config_name="baseline")
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

    done = load_done(jsonl_path) if cfg.inference.resume else {}
    print(f"model: {cfg.model.title}")
    print(f"model id: {cfg.model.model_id}")
    print(f"output dir: {output_dir}")
    print(f"test samples: {len(samples)}")
    print(f"already predicted: {len(done)}")

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
        desc="building-count test",
        unit="sample",
    )

    for batch in batched(pending, cfg.inference.batch_size):
        batch_started = time.time()
        batch_indices = [index for index, _ in batch]
        batch_samples = [sample for _, sample in batch]
        batch_outputs = infer_batch(
            model=model,
            processor=processor,
            samples=batch_samples,
            question=cfg.question,
            max_new_tokens=cfg.inference.max_new_tokens,
        )
        elapsed_per_sample = (time.time() - batch_started) / len(batch)

        for index, sample, (raw_output, pred) in zip(
            batch_indices,
            batch_samples,
            batch_outputs,
        ):
            gt = int(sample.answer)
            row = {
                "index": index,
                "model": cfg.model.key,
                "model_id": cfg.model.model_id,
                "image_a": sample.image_a,
                "image_b": sample.image_b,
                "gt": gt,
                "pred": pred,
                "correct": pred == gt,
                "bucket": bucket_name(gt),
                "raw_output": raw_output,
                "elapsed_sec": round(elapsed_per_sample, 4),
            }
            append_jsonl(jsonl_path, row)
            done[index] = row

        total_elapsed = time.time() - started
        progress.update(len(batch))
        progress.set_postfix(
            done=f"{len(done)}/{len(samples)}",
            elapsed=f"{total_elapsed / 60:.1f}m",
        )

    progress.close()

    print(f"saved: {jsonl_path}")


if __name__ == "__main__":
    main()
