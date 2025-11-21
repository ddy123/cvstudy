import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2

# 1. 加载处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
)

# 读取图像
frame = cv2.imread("/home/ddy/code/python/cvstudy/rose.png")
pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# 定义提示词列表
prompts = [
    "列出图像中所有可见的物体:",
    "图像中包含以下物体:",
    "详细描述图像中的每个物体:"
]

# 逐个处理每个提示词
for prompt in prompts:
    try:
        # 对每个提示词单独处理
        inputs = processor(pil_image, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        print(f"提示: {prompt}")
        print(f"描述: {description}")
        print(f"生成的描述: {description}\n")
    except Exception as e:
        print(f"处理提示 '{prompt}' 时出错: {e}")