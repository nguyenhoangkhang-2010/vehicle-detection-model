from ultralytics import YOLO
"""Main train model"""
"""
def train():
    model = YOLO("yolo26n.pt") 

    model.train(
        data="data/processed/data.yaml", 
        epochs=100,                    
        imgsz=640,                   
        device='cpu',    
        project=r"D:\vehicle-identification-model\outputs",  
        name="vehicle_detection_v1" 
    )

if __name__ == "__main__":
    train()
"""

"""Train tiếp các epoch
def resume_train():
    model = YOLO("outputs/vehicle_detection_v1/weights/last.pt")

    model.train(resume=True)

if __name__ == "__main__":
    resume_train()
    """
    
#Load lại model đã train và test thử trên Anaconda
"""
model = YOLO('outputs/vehicle_detection_v1/weights/best.pt') 

results = model('test.jpg', save=True, show=True, project='results', name='predict')
"""