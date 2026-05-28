import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util import (
    build_prompt_with_images,
    configure_generation_tokens,
    move_to_model_device,
    repo_path,
    suppress_http_logs,
)

MODEL_ID = "OpenGVLab/InternVL3_5-1B-HF"
IMAGE_PATH = repo_path("test/test.png")
PROMPT = "Describe this image in detail."

print("CUDA available:", torch.cuda.is_available())
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA GPU is not available.")

suppress_http_logs()

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()
configure_generation_tokens(model, processor)

print("Model loaded.")
print("Model device:", model.device)

image = Image.open(IMAGE_PATH).convert("RGB")
text = build_prompt_with_images(processor, [image], PROMPT)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
)
inputs = move_to_model_device(inputs, model)

print("Running inference...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )

generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

print("CUDA allocated after inference:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("CUDA reserved after inference:", torch.cuda.memory_reserved() / 1024**3, "GB")

print("\n===== Response =====")
print(generated_text)
