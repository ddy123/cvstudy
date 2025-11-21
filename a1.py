import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from collections import defaultdict
import time

class FlavorObjectDetector:
    def __init__(self):
        # Initialize BLIP model for image captioning (tiny vision-language model)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
    use_safetensors=True)
       
        # Define flavor-related objects
        self.flavor_objects = {
            'flowers': ['rose', 'lily', 'jasmine', 'lavender', 'orchid', 'daisy', 'sunflower',
                       'tulip', 'gardenia', 'honeysuckle', 'chrysanthemum', 'peony', 'carnation'],
            'fruits': ['apple', 'orange', 'lemon', 'banana', 'strawberry', 'grape', 'pineapple',
                      'mango', 'peach', 'pear', 'cherry', 'watermelon', 'blueberry', 'raspberry'],
            'herbs': ['mint', 'basil', 'rosemary', 'thyme', 'sage', 'cinnamon', 'vanilla'],
            'foods': ['coffee', 'tea', 'chocolate', 'bread', 'cheese']
        }
       
        self.detected_objects = defaultdict(list)
       
    def analyze_frame(self, frame, timestamp):
        """Analyze a single frame for flavor-generating objects"""
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
       
        # Generate caption using BLIP model
        inputs = self.processor(pil_image, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=50, num_beams=5)
        caption = self.processor.decode(out[0], skip_special_tokens=True).lower()
       
        # Check for flavor objects in caption
        detected_in_frame = []
        for category, objects in self.flavor_objects.items():
            for obj in objects:
                if obj in caption:
                    detected_in_frame.append((obj, category))
       
        return detected_in_frame
   
    def process_video(self, video_path, frame_interval=30):
        """Process video and detect flavor objects with timestamps"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
       
        print("Processing video for flavor objects...")
       
        while True:
            ret, frame = cap.read()
            if not ret:
                break
               
            # Process every nth frame to optimize performance
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
               
                # Analyze frame
                detected_objects = self.analyze_frame(frame, timestamp)
               
                # Update tracking
                for obj, category in detected_objects:
                    if not self.detected_objects[obj] or timestamp - self.detected_objects[obj][-1]['end_time'] > 2.0:
                        # New appearance
                        self.detected_objects[obj].append({
                            'start_time': timestamp,
                            'end_time': timestamp,
                            'category': category
                        })
                    else:
                        # Update end time for continuous appearance
                        self.detected_objects[obj][-1]['end_time'] = timestamp
               
                print(f"Frame {frame_count}, Time: {timestamp:.2f}s - Detected: {detected_objects}")
           
            frame_count += 1
       
        cap.release()
        return self.cleanup_detections()
   
    def cleanup_detections(self):
        """Clean up and format detection results"""
        results = []
        for obj, appearances in self.detected_objects.items():
            for appearance in appearances:
                results.append({
                    'object': obj,
                    'category': appearance['category'],
                    'start_time': appearance['start_time'],
                    'end_time': appearance['end_time'],
                    'duration': appearance['end_time'] - appearance['start_time']
                })
       
        # Sort by start time
        results.sort(key=lambda x: x['start_time'])
        return results
   
    def format_time(self, seconds):
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

def main():
    # Initialize detector
    detector = FlavorObjectDetector()
   
    # Process video
    video_path = "/home/ddy/code/python/cvstudy/input.mp4"  # Replace with your video path
    results = detector.process_video(video_path, frame_interval=30)
   
    # Display results
    print("\n" + "="*60)
    print("FLAVOR OBJECT DETECTION RESULTS")
    print("="*60)
   
    for result in results:
        print(f"Object: {result['object'].title()} ({result['category']})")
        print(f"Appears: {detector.format_time(result['start_time'])}")
        print(f"Disappears: {detector.format_time(result['end_time'])}")
        print(f"Duration: {result['duration']:.1f} seconds")
        print("-" * 40)

if __name__ == "__main__":
    main()
