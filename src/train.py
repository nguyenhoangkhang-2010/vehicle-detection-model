from ultralytics import YOLO
import os

model = YOLO("yolo12n.pt")

model.train(
    data="data.yaml", 
    epochs = 100, 
    imgsz = 640, 
    name= 'trafficv1',
    batch= 16,
    patience= 20,
    device = 0
    )

results = model("path/processed", save= True,  conf= 0.25)