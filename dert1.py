import os
import cv2
import numpy as np
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from collections import defaultdict

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class AdvancedFlavorDetector:
    def __init__(self):
        print("正在加载模型...")
        # BLIP for captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
                                                                       use_safetensors=True).to(device)
       
        # DETR for object detection
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device) 
        print("模型加载完成")
   
    def detect_objects_detr(self, image, threshold=0.5):
        """使用 DETR 进行目标检测"""
        try:
            # 确保图像是 PIL 格式
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
                
            print(f"图像尺寸: {pil_image.size}")
            
            # 预处理
            inputs = self.detr_processor(images=pil_image, return_tensors="pt").to(device)
            
            # 模型推理
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
           
            # 后处理 - 将 target_sizes 也移到设备上
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )[0]
           
            detected_items = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                item_name = self.detr_model.config.id2label[label.item()]
                detected_items.append({
                    "label": item_name,
                    "score": score.item(),
                    "box": [round(coord.item(), 2) for coord in box]
                })
                print(f"检测到: {item_name}, 置信度: {score.item():.4f}")
           
            return detected_items
        except Exception as e:
            print(f"DETR检测错误: {str(e)}")
            return []
   
    def analyze_frame_advanced(self, frame):
        """使用 BLIP 和 DETR 进行高级帧分析"""
        try:
            # 转换为PIL图像
            if isinstance(frame, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                pil_image = frame
                
            print(f"分析图像尺寸: {pil_image.size}")
            
            # Method 1: BLIP 图像描述
            print("使用 BLIP 生成图像描述...")
            inputs = self.blip_processor(pil_image, return_tensors="pt").to(device)
            out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            print(f"BLIP 描述: {caption}")
           
            # Method 2: DETR 目标检测
            print("使用 DETR 进行目标检测...")
            
            # 尝试不同的阈值
            thresholds = [0.7, 0.5, 0.3]
            detected_objects = []
            
            for threshold in thresholds:
                print(f"尝试阈值: {threshold}")
                detected_objects = self.detect_objects_detr(pil_image, threshold)
                if detected_objects:
                    print(f"使用阈值 {threshold} 检测到 {len(detected_objects)} 个物体")
                    break
                else:
                    print(f"使用阈值 {threshold} 未检测到物体")
            
            return detected_objects
            
        except Exception as e:
            print(f"分析帧错误: {str(e)}")
            return []

def test_with_local_images(detector, image_paths):
    """测试多张本地图片"""
    for image_path in image_paths:
        print(f"\n{'='*50}")
        print(f"测试图片: {image_path}")
        print(f"{'='*50}")
        
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            continue
            
        try:
            # 使用 PIL 打开图片
            image = Image.open(image_path).convert('RGB')
            print(f"图片尺寸: {image.size}")
            
            # 进行目标检测
            detected_objects = detector.analyze_frame_advanced(image)
            
            if detected_objects:
                print(f"检测结果: {[obj['label'] for obj in detected_objects]}")
            else:
                print("未检测到任何物体")
                
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")

if __name__ == '__main__':
    # 初始化检测器
    detector = AdvancedFlavorDetector()
    
    # 测试图片列表 - 修改为你的本地图片路径
    test_images = [
        '/home/ddy/code/python/cvstudy/orange.jpg',
        # 添加更多测试图片
        # '/path/to/your/image1.jpg',
        # '/path/to/your/image2.jpg',
    ]
    
    # 测试本地图片
    test_with_local_images(detector, test_images)
    
    # 交互式测试
    while True:
        user_input = input("\n输入图片路径进行测试 (或输入 'quit' 退出): ").strip()
        if user_input.lower() == 'quit':
            break
        
        if os.path.exists(user_input):
            try:
                image = Image.open(user_input).convert('RGB')
                detected_objects = detector.analyze_frame_advanced(image)
                
                if detected_objects:
                    print(f"检测结果: {[obj['label'] for obj in detected_objects]}")
                else:
                    print("未检测到任何物体")
            except Exception as e:
                print(f"处理图片时出错: {str(e)}")
        else:
            print(f"文件不存在: {user_input}")