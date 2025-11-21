import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
from collections import defaultdict
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AdvancedFlavorDetector:
    def __init__(self):
        # BLIP for captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
    use_safetensors=True).to(device)
       
        # DETR for object detection (optional enhancement)
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
       
        self.flavor_objects = {
            'flowers': ['rose', 'lily', 'jasmine', 'lavender', 'orchid', 'daisy', 'sunflower',
                       'tulip', 'gardenia', 'honeysuckle', 'chrysanthemum', 'peony', 'carnation',
                       'flower', 'blossom', 'bloom'],
            'fruits': ['apple', 'orange', 'lemon', 'banana', 'strawberry', 'grape', 'pineapple',
                      'mango', 'peach', 'pear', 'cherry', 'watermelon', 'blueberry', 'raspberry',
                      'fruit', 'berry'],
            'herbs': ['mint', 'basil', 'rosemary', 'thyme', 'sage', 'cinnamon', 'vanilla', 'herb'],
            'foods': ['coffee', 'tea', 'chocolate', 'bread', 'cheese', 'wine', 'beer', 'perfume']
        }
       
        self.detected_objects = defaultdict(list)
   
    def detect_objects_detr(self, image):
        """Use DETR for more accurate object detection"""
        inputs = self.detr_processor(images=image, return_tensors="pt").to(device)
        outputs = self.detr_model(**inputs)
       
        # Convert outputs to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
       
        detected_items = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detected_items.append(self.detr_model.config.id2label[label.item()])
       
        return detected_items
   
    def analyze_frame_advanced(self, frame, timestamp):
        """Enhanced frame analysis using both BLIP and DETR"""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
       
        # Method 1: BLIP captioning
        inputs = self.blip_processor(pil_image, return_tensors="pt").to(device)
        out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True).lower()
       
        # Method 2: DETR object detection (optional)
        # detected_objects = self.detect_objects_detr(pil_image)
        # combined_text = caption + " " + " ".join(detected_objects)
       
        detected_in_frame = []
        for category, objects in self.flavor_objects.items():
            for obj in objects:
                if obj in caption:  # or obj in combined_text for enhanced version
                    detected_in_frame.append((obj, category))
       
        return detected_in_frame
   
    def process_video(self, video_path, frame_interval=30):
        """Process video with enhanced detection"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
       
        print("Processing video for flavor objects...")
       
        while True:
            ret, frame = cap.read()
            if not ret:
                break
               
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                detected_objects = self.analyze_frame_advanced(frame, timestamp)
               
                # Track object appearances
                for obj, category in detected_objects:
                    self.track_object(obj, category, timestamp)
           
            frame_count += 1
       
        cap.release()
        return self.cleanup_detections()
   
    def track_object(self, obj, category, timestamp):
        """Track object appearances and disappearances"""
        if not self.detected_objects[obj] or timestamp - self.detected_objects[obj][-1]['end_time'] > 3.0:
            # New appearance (gap > 3 seconds)
            self.detected_objects[obj].append({
                'start_time': timestamp,
                'end_time': timestamp,
                'category': category
            })
        else:
            # Update end time for continuous appearance
            self.detected_objects[obj][-1]['end_time'] = timestamp
   
    def cleanup_detections(self):
        """Final processing of detection results"""
        results = []
        for obj, appearances in self.detected_objects.items():
            for appearance in appearances:
                if appearance['end_time'] - appearance['start_time'] >= 1.0:  # Minimum 1 second duration
                    results.append({
                        'object': obj,
                        'category': appearance['category'],
                        'start_time': appearance['start_time'],
                        'end_time': appearance['end_time'],
                        'duration': appearance['end_time'] - appearance['start_time']
                    })
       
        results.sort(key=lambda x: x['start_time'])
        return results
   
    def format_time(self, seconds):
        """Format seconds to MM:SS.mmm"""
        minutes = int(seconds // 60)
        seconds_remaining = seconds % 60
        return f"{minutes:02d}:{seconds_remaining:06.3f}"

# Usage example
if __name__ == "__main__":
    detector = AdvancedFlavorDetector()
    results = detector.process_video("/home/ddy/code/python/cvstudy/input.mp4")
   
    print("\nFLAVOR OBJECT TIMELINE:")
    print("="*70)
    for result in results:
        print(f"📍 {result['object'].upper():<15} | "
              f"Category: {result['category']:<10} | "
              f"Time: {detector.format_time(result['start_time'])} - {detector.format_time(result['end_time'])} | "
              f"Duration: {result['duration']:.1f}s")   