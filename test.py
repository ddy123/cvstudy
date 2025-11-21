""" import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")    """

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torchvision
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import defaultdict

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# 强制使用 safetensors 格式（如果模型支持）
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True # 强制使用 safetensors
    )
    print("BLIP 模型加载成功！")
except Exception as e:
    print(f"错误: {e}")
