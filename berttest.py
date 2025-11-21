import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
from PIL import Image
import requests
from transformers import pipeline,BlipProcessor, BlipForConditionalGeneration

# 1. 加载处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
)
bert_classifier = pipeline("text-classification", 
  model="bert-base-uncased")

object_keywords = {
            'fruit': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'fruit'],
            'flower': ['flower', 'rose', 'tulip', 'sunflower', 'blossom'],
            'vegetable': ['carrot', 'broccoli', 'tomato', 'potato', 'vegetable']
        }

# 2. 准备图像
url = "/home/ddy/code/python/cvstudy/rose.png"
image=cv2.imread(url)
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 使用特定的提示词引导BLIP生成更具体的描述
prompts = [
    "列出图像中所有可见的物体:",
    "图像中包含以下物体:",
    "详细描述图像中的每个物体:"
]

for prompt in prompts:
    # 3. 使用处理器预处理图像
    #inputs = processor(image, return_tensors="pt")  # 返回PyTorch张量
    inputs = processor(pil_image, prompts,return_tensors="pt")
    # 4. 生成描述
    outputs = model.generate(**inputs)

    # 5. 解码输出
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    #print(f"生成的描述: {caption}")
    print(f"提示: {prompt}")
    print(f"描述: {caption}\n")

# 步骤2: 使用BERT对描述文本进行气味分类
#scent_results = bert_classifier(caption)
    
    # 提取气味信息
""" scents = []
for result in scent_results:
    if result['score'] > 0.5:  # 设置置信度阈值
        scents.append(result['label'])
for result in scent_results:
    print(result)  """


""" detected_objects = []
for obj_type, keywords in object_keywords.items():
    for keyword in keywords:
        if keyword.lower() in caption.lower():
                detected_objects.append(obj_type)
        break  # 找到一种就跳出内层循环

print(detected_objects)  """