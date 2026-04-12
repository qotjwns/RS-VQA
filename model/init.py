from transformers import AutoModel, AutoTokenizer
import torch

model_name = "OpenGVLab/InternVL3_5-8B"
cache_dir = r"D:\models\huggingface"

model = AutoModel.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    cache_dir=cache_dir,
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False,
    cache_dir=cache_dir,
)