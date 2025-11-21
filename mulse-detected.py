from flask import Flask, request, jsonify,send_file
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
import numpy as np
import io
from PIL import Image
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)

class AdvancedFlavorDetector:
    def __init__(self):
        # BLIP for captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
    use_safetensors=True).to(device)
       
        # DETR for object detection (optional enhancement)
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device) 
   
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
   
    def analyze_frame_advanced(self, frame):
        """Enhanced frame analysis using both BLIP and DETR"""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Method 1: BLIP captioning
        inputs = self.blip_processor(pil_image, return_tensors="pt").to(device)
        out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True).lower() 
       
        # Method 2: DETR object detection (optional)
        detected_objects = self.detect_objects_detr(pil_image)
        print(detected_objects)
        return detected_objects


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    print(request.method+'\n')
    print(request.url+'\n')
    image_data = request.get_data()
    #print(image_data)
    # 解码图像
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
    """ cv2.imshow('YOLO', image) 
    cv2.waitKey()  """
    print(request.headers.get('X-File-Name'))
    detector.analyze_frame_advanced(image)
    return jsonify({"success": True}),200

if __name__ == '__main__':
    #app.run(debug=True)
    detector = AdvancedFlavorDetector()
    app.run(debug=True, host='222.20.126.228', port=5002)
    