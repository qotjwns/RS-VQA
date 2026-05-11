import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_PATH = "./models/InternVL3_5-8B-HF"
IMAGE_PATH = "./test.jpg"
QUESTION = "Describe this image."

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
).eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": IMAGE_PATH},
            {"type": "text", "text": QUESTION},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=256)

answer_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
answer = processor.decode(answer_ids, skip_special_tokens=True)

print(answer.strip())
