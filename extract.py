import cv2
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torchvision
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import defaultdict


def extract_frames_basic(video_path, output_dir, interval=100):
    """
    基础帧提取函数
    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param interval: 帧间隔（每几帧提取一次）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"视频信息: {total_frames}帧, {fps:.2f}FPS, {duration:.2f}秒")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # 按间隔保存帧
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            # 3. 使用处理器预处理图像
            #inputs = processor(frame, return_tensors="pt")  # 返回PyTorch张量
            # 4. 生成描述
            #outputs = model.generate(**inputs)
            # 5. 解码输出
            #caption = processor.decode(outputs[0], skip_special_tokens=True)
            #print(f"生成的描述: {caption}")
            #cv2.imshow("image",frame)
            #cv2.waitKey()
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"已保存 {saved_count} 帧...")
        
        frame_count += 1
    
    cap.release()
    print(f"完成！共提取 {saved_count} 帧")

# 1. 加载处理器和模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
)    
# 使用示例
extract_frames_basic("/home/ddy/code/python/cvstudy/input.mp4", "/home/ddy/code/python/cvstudy/output_frames")