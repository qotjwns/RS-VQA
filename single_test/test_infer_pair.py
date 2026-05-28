from __future__ import annotations

import sys
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    logging as transformers_logging,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util import (
    build_prompt_with_images,
    configure_generation_tokens,
    load_test_building_count,
    move_to_model_device,
    parse_first_int,
    repo_path,
    suppress_http_logs,
)


# Test-only defaults. All paths are resolved from the repository root.
MODEL_ID = "OpenGVLab/InternVL3_5-1B-HF"
SAMPLE_INDEX = 300
MAX_NEW_TOKENS = 128
ANNOTATION_PATH = Path("data/coding/muti_task_data/test_task_data/count_build.json")
LOCAL_DATA_ROOT = Path("data")
VIS_SAVE_PATH = Path("outputs/pair_debug/test_infer_pair.png")
PAIR_PROMPT = (
    "How many buildings have been changed in these two remote sensing images? "
    "Answer with only one integer."
)


transformers_logging.set_verbosity_error()


def infer_pair_count(
    model,
    processor,
    image_a: Image.Image,
    image_b: Image.Image,
    question: str,
) -> tuple[str, int | None]:
    prompt = build_prompt_with_images(processor, [image_a, image_b], question)
    inputs = processor(
        text=[prompt],
        images=[[image_a, image_b]],
        return_tensors="pt",
        padding=True,
    )

    inputs = move_to_model_device(inputs, model)
    input_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=model.generation_config.pad_token_id,
        )

    if input_length and generated_ids.shape[-1] > input_length:
        generated_ids = generated_ids[:, input_length:]

    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    pred = parse_first_int(raw_output)
    if pred is not None and pred < 0:
        pred = None
    return raw_output, pred


def save_pair_figure(
    image_a: Image.Image,
    image_b: Image.Image,
    save_path: Path,
    sample_index: int,
    gt_answer: int | None,
    pred_answer: int | None,
    raw_output: str,
) -> tuple[Path, Path]:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_a, ax_b, ax_text = axes

    ax_a.imshow(image_a)
    ax_a.set_title("T1 (Image A)")
    ax_a.axis("off")

    ax_b.imshow(image_b)
    ax_b.set_title("T2 (Image B)")
    ax_b.axis("off")

    wrapped_output = textwrap.fill(raw_output, width=45) if raw_output else "(empty output)"
    ax_text.axis("off")
    ax_text.set_title("Model Output")
    ax_text.text(
        0.0,
        1.0,
        (
            f"gt: {gt_answer}\n"
            f"pred: {pred_answer}\n\n"
            f"raw_output:\n{wrapped_output}"
        ),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle(
        f"sample={sample_index} | gt_answer={gt_answer} | pred_answer={pred_answer}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path.suffix.lower() == ".svg":
        svg_path = save_path
        png_path = save_path.with_suffix(".png")
    else:
        png_path = save_path
        svg_path = save_path.with_suffix(".svg")

    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    suppress_http_logs()

    annotation_path = repo_path(ANNOTATION_PATH)
    data_root = repo_path(LOCAL_DATA_ROOT)
    vis_save_path = repo_path(VIS_SAVE_PATH)

    samples = load_test_building_count(
        annotation_path=annotation_path,
        data_root=data_root,
    )
    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= len(samples):
        raise IndexError(f"SAMPLE_INDEX {SAMPLE_INDEX} out of range (total={len(samples)})")

    sample = samples[SAMPLE_INDEX]
    gt_raw_answer = str(sample.answer).strip()
    gt_answer = parse_first_int(gt_raw_answer)
    image_a_path = Path(sample.image_a)
    image_b_path = Path(sample.image_b)

    print(f"model_id: {MODEL_ID}")
    print(f"sample_index: {SAMPLE_INDEX}")
    print(f"annotation_path: {annotation_path}")
    print(f"data_root: {data_root}")
    print(f"image_a: {image_a_path}")
    print(f"image_b: {image_b_path}")
    print(f"prompt_question: {PAIR_PROMPT}")
    print(f"gt_raw_answer: {gt_raw_answer}")
    print(f"gt(parsed): {gt_answer}")
    print(f"vis_save_path: {vis_save_path}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        tie_word_embeddings=False,
        trust_remote_code=True,
    ).eval()
    configure_generation_tokens(model, processor)

    image_a = Image.open(image_a_path).convert("RGB")
    image_b = Image.open(image_b_path).convert("RGB")

    raw_output, pred_answer = infer_pair_count(
        model=model,
        processor=processor,
        image_a=image_a,
        image_b=image_b,
        question=PAIR_PROMPT,
    )

    print("\n===== MODEL OUTPUT =====")
    print(raw_output)
    print(f"pred(parsed): {pred_answer}")

    png_path, svg_path = save_pair_figure(
        image_a=image_a,
        image_b=image_b,
        save_path=vis_save_path,
        sample_index=SAMPLE_INDEX,
        gt_answer=gt_answer,
        pred_answer=pred_answer,
        raw_output=raw_output,
    )
    print(f"saved visualization (png): {png_path}")
    print(f"saved visualization (svg): {svg_path}")


if __name__ == "__main__":
    main()
