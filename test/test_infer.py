import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# 경로 설정
MODEL_ID = "OpenGVLab/InternVL3_5-1B-HF"
IMAGE_PATH = "/workspace/RS-VQA/test/test.png"
PROMPT = "Describe this image in detail."

# CUDA 확인
print("CUDA available:", torch.cuda.is_available())
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA GPU is not available.")

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

print("Model loaded.")
print("Model device:", model.device)

# 이미지 로드
image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
)

inputs = {
    k: v.to(model.device) if torch.is_tensor(v) else v
    for k, v in inputs.items()
}

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