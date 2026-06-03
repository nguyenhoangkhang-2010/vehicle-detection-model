# src/detect.py
import numpy as np
from ultralytics import YOLO
from src.config import CLASS_NAMES

class YOLODetector:
    def __init__(self, model_path, imgsz=640, conf_threshold=0.45):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold

    def extract_boxes(self, frame):
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf_threshold, iou=0.65, verbose=False)[0]
        detections = []
        
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, clss):
                if cls_id in CLASS_NAMES:
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1
                    detections.append(([x1, y1, w, h], conf, cls_id))
                    
        return detections