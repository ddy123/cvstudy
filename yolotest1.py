import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import time

class LocalImageScentDetector:
    def __init__(self, model_path="/home/ddy/code/python/cvstudy/yolov8m.pt"):
        """
        初始化单张图片检测器，使用本地模型文件
        
        Args:
            model_path: 本地YOLO模型文件路径
        """
        print("🚀 初始化 YOLO 物体检测器...")
        print(f"📁 使用本地模型: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载本地模型
        self.model = YOLO(model_path)
        
        # 物体到气味的映射
        self.scent_mapping = {
            # 植物相关
            'potted plant': ['植物清香', '绿叶气息'],
            'vase': ['花香', '植物芳香'],
            
            # 水果
            'apple': ['苹果香', '果香'],
            'orange': ['橙子香', '柑橘调'],
            'banana': ['香蕉味', '甜香'],
            
            # 食物
            'sandwich': ['面包香', '食材混合香'],
            'pizza': ['烘焙香', '奶酪香'],
            'cake': ['甜点香', '糖霜香'],
            'hot dog': ['烤肉香', '面包香'],
            
            # 饮品相关
            'wine glass': ['葡萄酒香', '果酒气息'],
            'cup': ['饮品香气', '热饮香'],
            'bottle': ['瓶中物气味', '液体香气'],
            'bowl': ['食物香气', '汤品香'],
            
            # 其他
            'book': ['书香', '纸张味'],
            'person': ['人体气息', '香水味'],
            'chair': ['木质调', '家具气息'],
            'dining table': ['木质调', '食物残留香'],
        }
        
        print("✅ 检测器初始化完成")
    
    def detect_objects(self, image_path, confidence_threshold=0.5):
        """
        检测单张图片中的物体
        
        Args:
            image_path: 图片路径
            confidence_threshold: 置信度阈值
        
        Returns:
            dict: 包含检测结果和气味信息的字典
        """
        # 检查图片是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        print(f"📷 图片尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 运行YOLO检测
        start_time = time.time()
        results = self.model(image, verbose=False)
        detection_time = time.time() - start_time
        
        # 解析检测结果
        detections = []
        all_scents = set()
        
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence > confidence_threshold:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # 获取气味信息
                    scents = self.scent_mapping.get(class_name, [])
                    
                    detection_info = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox.astype(int),
                        'scents': scents
                    }
                    
                    detections.append(detection_info)
                    all_scents.update(scents)
        
        return {
            'image': image,
            'detections': detections,
            'scents': list(all_scents),
            'detection_time': detection_time,
            'total_objects': len(detections)
        }
    
    def draw_detections(self, image, detections):
        """
        在图片上绘制检测结果
        """
        result_image = image.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            scents = detection['scents']
            
            # 绘制边界框
            color = self._get_color(i)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            label = f"{class_name} {confidence:.2f}"
            
            # 计算标签背景尺寸
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 绘制标签背景
            cv2.rectangle(result_image, 
                         (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), 
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(result_image, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 绘制气味信息（如果框足够大）
            if scents and (y2 - y1) > 60:
                scent_text = f"Scent: {scents[0]}"
                cv2.putText(result_image, scent_text, 
                           (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result_image
    
    def _get_color(self, index):
        """根据索引生成不同的颜色"""
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
        ]
        return colors[index % len(colors)]
    
    def analyze_image(self, image_path, output_path=None, confidence=0.5, show_result=True):
        """
        分析单张图片并显示/保存结果
        """
        print(f"\n🎯 开始分析图片: {image_path}")
        print("=" * 50)
        
        try:
            # 检测物体
            result_info = self.detect_objects(image_path, confidence)
            
            # 绘制检测结果
            image_with_detections = self.draw_detections(
                result_info['image'], 
                result_info['detections']
            )
            
            # 显示结果
            if show_result:
                # 调整显示大小（如果图片太大）
                display_image = self._resize_for_display(image_with_detections, max_width=1200)
                cv2.imshow('YOLO物体检测 - 气味识别', display_image)
                print("👀 结果显示中... 按任意键关闭窗口")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # 保存结果
            if output_path:
                cv2.imwrite(output_path, image_with_detections)
                print(f"💾 结果已保存: {output_path}")
            
            # 打印详细结果
            self._print_detailed_results(result_info)
            
            return result_info
            
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            return None
    
    def _resize_for_display(self, image, max_width=1200):
        """调整图片大小以便显示"""
        height, width = image.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        return image
    
    def _print_detailed_results(self, result_info):
        """打印详细检测结果"""
        print("\n📊 检测结果详情:")
        print("-" * 40)
        print(f"检测时间: {result_info['detection_time']*1000:.1f}ms")
        print(f"检测物体总数: {result_info['total_objects']}")
        
        if result_info['detections']:
            print("\n🔍 检测到的物体:")
            for i, detection in enumerate(result_info['detections'], 1):
                print(f"  {i}. {detection['class_name']} "
                      f"(置信度: {detection['confidence']:.2f})")
                if detection['scents']:
                    print(f"     气味: {', '.join(detection['scents'])}")
        
        if result_info['scents']:
            print(f"\n👃 检测到的气味: {', '.join(result_info['scents'])}")
        else:
            print(f"\n👃 未检测到已知气味")
        
        print("=" * 50)

# 简易使用函数
def quick_detect(image_path, model_path="yolov8m.pt", confidence=0.5):
    """
    快速检测函数 - 一行代码即可使用
    
    Args:
        image_path: 图片路径
        model_path: 模型路径，默认为当前目录的yolov8m.pt
        confidence: 置信度阈值
    """
    detector = LocalImageScentDetector(model_path)
    return detector.analyze_image(image_path, confidence=confidence)

# 主函数
def main():
    """主函数 - 使用示例"""
    # 创建检测器
    detector = LocalImageScentDetector("yolov8m.pt")
    
    # 示例图片路径（请替换为您自己的图片路径）
    test_images = [
        "/home/ddy/code/python/cvstudy/rose.png"
    ]
    
    # 查找存在的图片
    image_path = None
    for img_path in test_images:
        if os.path.exists(img_path):
            image_path = img_path
            print(f"✅ 找到图片: {img_path}")
            break
    
    if image_path is None:
        print("❌ 未找到测试图片")
        print("💡 请将图片放在当前目录下，或直接指定图片路径")
        
        # 让用户输入图片路径
        image_path = input("请输入图片路径: ").strip()
        if not os.path.exists(image_path):
            print("❌ 指定的图片路径不存在")
            return
    
    # 分析图片
    result = detector.analyze_image(
        image_path=image_path,
        output_path="detection_result.jpg",  # 保存结果
        confidence=0.5,  # 置信度阈值
        show_result=True  # 显示结果
    )
    
    if result:
        print("🎉 图片分析完成！")

if __name__ == "__main__":
    main()