import argparse
import json
import os
import re
import time
from pathlib import Path
from threading import Thread

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CDVQA_DIR = DATA_DIR / "CDVQA-main"
SECOND_DIR = DATA_DIR / "SECOND_train_set"
TEST_DIR = DATA_DIR / "test"

MODEL_NAME = "OpenGVLab/InternVL3_5-14B"
CACHE_DIR = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CDVQA Test inference.")
    parser.add_argument("--question-id", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--start-image-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--local-files-only", type=bool, default=False)
    return parser.parse_args()


def build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_pair_image(file_name: str, input_size: int = 448) -> torch.Tensor:
    im1_path = SECOND_DIR / "im1" / file_name
    im2_path = SECOND_DIR / "im2" / file_name
    if not im1_path.exists() or not im2_path.exists():
        raise FileNotFoundError(f"Missing pair image: {im1_path}, {im2_path}")

    im1 = Image.open(im1_path).convert("RGB")
    im2 = Image.open(im2_path).convert("RGB")

    canvas = Image.new("RGB", (im1.width + im2.width, max(im1.height, im2.height)))
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (im1.width, 0))

    transform = build_transform(input_size)
    return transform(canvas).unsqueeze(0)


def run_with_tqdm(fn, desc="Inference", refresh=0.1, extend_step=5.0):
    result = {}
    error = {}

    def target():
        try:
            with torch.inference_mode():
                result["value"] = fn()
        except Exception as exc:
            error["value"] = exc

    thread = Thread(target=target, daemon=True)
    thread.start()

    start = time.perf_counter()
    last = start

    with tqdm(total=extend_step, desc=desc, unit="s", dynamic_ncols=True) as pbar:
        while thread.is_alive():
            time.sleep(refresh)
            now = time.perf_counter()
            delta = now - last
            pbar.update(delta)
            last = now

            if pbar.total - pbar.n < 1.0:
                pbar.total += extend_step
                pbar.refresh()

        now = time.perf_counter()
        if now > last:
            pbar.update(now - last)

        pbar.total = max(pbar.total, pbar.n)
        pbar.refresh()

    thread.join()
    if "value" in error:
        raise error["value"]

    print(f"\nInference time: {time.perf_counter() - start:.2f} sec")
    return result["value"]


def build_choices_by_type(test_questions: list, test_answers: list) -> dict:
    question_type_by_id = {q["id"]: q["type"] for q in test_questions}
    choices_by_type = {}
    for a in test_answers:
        qtype = question_type_by_id[a["question_id"]]
        choices_by_type.setdefault(qtype, set()).add(a["answer"])
    return {k: sorted(v) for k, v in choices_by_type.items()}


def normalize_prediction(pred: str, choices: list[str]) -> str:
    text = pred.strip()
    lower = text.lower()
    choice_map = {c.lower(): c for c in choices}

    if lower in choice_map:
        return choice_map[lower]

    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    for tok in tokens:
        if tok in choice_map:
            return choice_map[tok]

    if tokens:
        joined = "_".join(tokens)
        if joined in choice_map:
            return choice_map[joined]

    return text


def main() -> None:
    args = parse_args()

    test_questions = json.loads((CDVQA_DIR / "Test_questions.json").read_text(encoding="utf-8"))["questions"]
    test_images = json.loads((CDVQA_DIR / "Test_images.json").read_text(encoding="utf-8"))["images"]
    test_answers = json.loads((CDVQA_DIR / "Test_answers.json").read_text(encoding="utf-8"))["answers"]

    question_map = {q["id"]: q for q in test_questions}
    image_map = {img["id"]: img for img in test_images}
    answer_map = {a["question_id"]: a["answer"] for a in test_answers}
    choices_by_type = build_choices_by_type(test_questions, test_answers)

    if args.question_id is not None:
        if args.question_id not in question_map:
            raise KeyError(f"question_id={args.question_id} not found in Test_questions.json")
        selected_question_ids = [args.question_id]
    else:
        candidate_images = [
            img for img in sorted(test_images, key=lambda x: x["id"]) if img["id"] >= args.start_image_id
        ][: args.num_images]
        selected_question_ids = []
        for img in candidate_images:
            qids = img.get("questions_ids", [])
            first_valid_qid = next((qid for qid in qids if qid in question_map), None)
            if first_valid_qid is not None:
                selected_question_ids.append(first_valid_qid)

        if not selected_question_ids:
            raise RuntimeError("No valid questions found for selected image range.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
        cache_dir=CACHE_DIR,
        local_files_only=args.local_files_only,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=CACHE_DIR,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total = 0
    matched = 0
    for idx, qid in enumerate(selected_question_ids, start=1):
        q = question_map[qid]
        img = image_map[q["img_id"]]
        file_name = img["file_name"]
        choices = choices_by_type[q["type"]]
        gt_answer = answer_map.get(q["id"])

        print(f"\n===== Sample {idx}/{len(selected_question_ids)} =====")
        print(f"question_id: {q['id']}")
        print(f"image_id   : {q['img_id']}")
        print(f"file_name  : {file_name}")
        print(f"q_type     : {q['type']}")
        print(f"question   : {q['question']}")
        print(f"choices    : {choices}")
        if gt_answer is not None:
            print(f"gt_answer  : {gt_answer}")

        pixel_values = load_pair_image(file_name).to(device=device, dtype=dtype)
        prompt = (
            "<image>\n"
            "The image is two remote sensing images stitched horizontally. "
            "Left is pre-event and right is post-event.\n"
            f"Question: {q['question']}\n"
            f"Choose exactly one answer from: {', '.join(choices)}\n"
            "Return only one label."
        )

        generation_config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        response = run_with_tqdm(
            lambda: model.chat(tokenizer, pixel_values, prompt, generation_config),
            desc=f"InternVL CDVQA sample {idx}",
        )
        normalized = normalize_prediction(response, choices)

        print("\n[Response]")
        print(response)
        print("\n[Normalized]")
        print(normalized)
        if gt_answer is not None:
            is_match = normalized == gt_answer
            print("\n[Match]")
            print(is_match)
            total += 1
            if is_match:
                matched += 1

    if total > 0:
        print("\n===== Summary =====")
        print(f"matched: {matched}/{total} ({matched / total:.2%})")


if __name__ == "__main__":
    main()
