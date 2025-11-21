import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. 加载处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
)

# 2. 准备图像
url = "/home/ddy/code/python/cvstudy/rose.png"
image = Image.open(url)

# 3. 使用处理器预处理图像
inputs = processor(image, return_tensors="pt")  # 返回PyTorch张量

# 4. 生成描述
outputs = model.generate(**inputs)

# 5. 解码输出
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(f"生成的描述: {caption}")