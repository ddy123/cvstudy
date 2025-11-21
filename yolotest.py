import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import os

model_path = os.path.expanduser("/home/ddy/code/python/cvstudy/yolov8m.pt")
model = YOLO(model_path)

class YOLOScentDetector:
    def __init__(self, model_size='m'):
        """
        初始化YOLO物体检测器
        model_size: n/s/m/l/x 对应不同模型大小
        """
        # 加载预训练YOLO模型
        self.model = YOLO(f'yolov8{model_size}.pt')
        #self.model = YOLO(model)
        
        # 物体到气味的映射
        self.scent_mapping = {
            # 植物相关
            'potted plant': ['植物清香', '绿叶气息'],
            'vase': ['花香', '植物芳香'],
            
            # 食物水果
            'apple': ['苹果香', '果香'],
            'orange': ['橙子香', '柑橘调'],
            'banana': ['香蕉味', '甜香'],
            'sandwich': ['面包香', '食材混合香'],
            'pizza': ['烘焙香', '奶酪香'],
            'cake': ['甜点香', '糖霜香'],
            
            # 饮品相关
            'wine glass': ['葡萄酒香', '果酒气息'],
            'cup': ['饮品香气', '热饮香'],
            'bottle': ['瓶中物气味', '液体香气'],
            
            # 其他
            'book': ['书香', '纸张味'],
            'candle': ['蜡香', '燃烧气息']
        }
        
        # 检测历史记录
        self.detection_history = defaultdict(list)
        self.frame_count = 0
        
    def detect_objects(self, frame, confidence_threshold=0.5):
        """
        使用YOLO检测物体
        """
        # 运行YOLO检测
        results = self.model(frame, verbose=False)
        
        detected_objects = []
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > confidence_threshold:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    detected_objects.append({
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': bbox,
                        'scent': self.scent_mapping.get(class_name, [])
                    })
        
        return detected_objects
    
    def draw_detections(self, frame, detections):
        """
        在帧上绘制检测结果
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox'].astype(int)
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签背景
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # 绘制标签文本
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # 如果有气味信息，显示在框上方
            if detection['scent']:
                scent_text = f"Scent: {', '.join(detection['scent'])}"
                cv2.putText(frame, scent_text, (x1, y1 - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame
    
    def process_video_realtime(self, video_source=0, output_file=None):
        """
        实时处理视频并检测物体和气味
        """
        # 打开视频源
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("❌ 无法打开视频源")
            return
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 设置视频写入器（如果需要保存结果）
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        print("🎬 开始实时物体检测...")
        print("按 'q' 退出，按 'p' 暂停")
        
        paused = False
        start_time = time.time()
        frame_count = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 物体检测
                detection_start = time.time()
                detections = self.detect_objects(frame)
                detection_time = time.time() - detection_start
                
                # 绘制检测结果
                frame_with_detections = self.draw_detections(frame.copy(), detections)
                
                # 添加信息面板
                frame_with_detections = self.add_info_panel(
                    frame_with_detections, detections, detection_time, frame_count
                )
                
                # 显示结果
                cv2.imshow('YOLO物体检测 - 气味识别', frame_with_detections)
                
                # 保存结果（如果需要）
                if output_file:
                    out.write(frame_with_detections)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸️ 暂停" if paused else "▶️ 继续")
        
        # 清理资源
        cap.release()
        if output_file:
            out.release()
        cv2.destroyAllWindows()
        
        # 性能统计
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\n📊 性能统计:")
        print(f"总帧数: {frame_count}")
        print(f"总时间: {total_time:.2f}s")
        print(f"平均FPS: {avg_fps:.2f}")
    
    def add_info_panel(self, frame, detections, detection_time, frame_count):
        """
        在帧上添加信息面板
        """
        # 基本信息
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detection Time: {detection_time*1000:.1f}ms", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Objects Detected: {len(detections)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 检测到的气味
        current_scents = set()
        for detection in detections:
            current_scents.update(detection['scent'])
        
        if current_scents:
            scent_text = "Detected Scents: " + ", ".join(current_scents)
            cv2.putText(frame, scent_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No scents detected", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame
    
    def analyze_video_scents(self, video_path, frame_interval=10):
        """
        分析视频中的气味出现模式
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        scent_timeline = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                # 检测物体
                detections = self.detect_objects(frame)
                
                # 收集气味信息
                current_scents = set()
                for detection in detections:
                    current_scents.update(detection['scent'])
                
                if current_scents:
                    scent_timeline.append({
                        'timestamp': timestamp,
                        'scents': list(current_scents),
                        'objects': [d['class_name'] for d in detections]
                    })
            
            frame_count += 1
        
        cap.release()
        return scent_timeline

# 🚀 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = YOLOScentDetector(model_size='m')
    
    # 方法1: 实时摄像头检测
    #print("📷 启动摄像头实时检测...")
    #detector.process_video_realtime(0)  # 0 表示默认摄像头
    
    # 方法2: 处理视频文件
    detector.process_video_realtime("/home/ddy/code/python/cvstudy/input.mp4", "/home/ddy/code/python/cvstudy/output_frames")
    
    # 方法3: 分析视频气味时间线
    # timeline = detector.analyze_video_scents('input_video.mp4')
    # for entry in timeline:
    #     print(f"Time: {entry['timestamp']:.1f}s - Scents: {entry['scents']}")