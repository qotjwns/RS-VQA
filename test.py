from pathlib import Path

local_model_dir = Path(r"D:\models\huggingface\models--OpenGVLab--InternVL3_5-14B")

print(local_model_dir.exists())      # True 여야 함
print(local_model_dir.is_dir())      # True 여야 함
print((local_model_dir / "config.json").exists())   # 보통 True 여야 함