import os
from dotenv import load_dotenv
from ultralytics import YOLO
load_dotenv()

if __name__ == '__main__':
    model_path = os.getenv("MODEL_TRAIN_BASE", "yolo26x-pose.pt")
    project_dir = os.getenv("TRAIN_PROJECT_DIR", "Largest")
    model = YOLO(model_path)

    results = model.train(
    data='judo_pose.yaml',
    epochs=150,          
    patience=20,         
    batch=16,             
    workers=4,
    imgsz=640,
    dropout=0.1,         
    augment=True,         
    project = project_dir,
    name = 'yolo26X-pose-judo'
)