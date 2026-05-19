from __future__ import annotations

import sys
import time
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util import (
    TestSample,
    append_jsonl,
    batched,
    bucket_name,
    build_prompt_with_images,
    configure_generation_tokens,
    index_rows,
    load_jsonl,
    load_test_building_count,
    move_to_model_device,
    parse_first_int,
    repo_path,
    suppress_http_logs,
)


transformers_logging.set_verbosity_error()


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
        build_prompt_with_images(processor, [image_a, image_b], question)
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


@hydra.main(version_base=None, config_path="configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    suppress_http_logs()

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
