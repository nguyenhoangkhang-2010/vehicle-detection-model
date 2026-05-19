from ultralytics import YOLO
from pathlib import Path
"""Main train model"""

ROOT = Path(__file__).resolve().parent

def train():
    model = YOLO("yolo26n.pt") 

    model.train(
        data="data/processed/data.yaml", 
        epochs=100,                    
        imgsz=640,                   
        device='cpu',    
        project=str(ROOT / "outputs"), 
        name="vehicle_detection_v1"
    )

if __name__ == "__main__":
    train()


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