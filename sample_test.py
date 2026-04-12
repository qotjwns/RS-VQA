import time
from threading import Thread

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

model_name = "OpenGVLab/InternVL3_5-14B"
cache_dir = r"D:\models\huggingface"
image_path = r"D:\rs_change_vqa\test.jpg"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0)   # [1, 3, H, W]
    return pixel_values

def run_with_tqdm(fn, desc="Inference", refresh=0.1, extend_step=5.0):
    result = {}
    error = {}

    def target():
        try:
            with torch.inference_mode():
                result["value"] = fn()
        except Exception as e:
            error["value"] = e

    thread = Thread(target=target, daemon=True)
    thread.start()

    start = time.perf_counter()
    last = start

    with tqdm(
        total=extend_step,
        desc=desc,
        unit="s",
        dynamic_ncols=True,
    ) as pbar:
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

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model = AutoModel.from_pretrained(
    model_name,
    dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto" if device == "cuda" else None,
    cache_dir=cache_dir,
    local_files_only=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False,
    cache_dir=cache_dir,
    local_files_only=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

pixel_values = load_image(image_path).to(device=device, dtype=dtype)

question = "<image>\n이 이미지에 무엇이 보이는지 자세히 설명해줘."
generation_config = {
    "max_new_tokens": 64,
    "do_sample": False,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

response = run_with_tqdm(
    lambda: model.chat(tokenizer, pixel_values, question, generation_config),
    desc="InternVL inference",
)

print("\n[Response]")
print(response)