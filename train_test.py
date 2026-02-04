from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('yolo26x-pose.pt')

    # 2. Train the model
    results = model.train(
    data='judo_pose.yaml',
    epochs=150,          # Reduced from 200 because it learns fast
    patience=20,         # STRICT early stopping. If it doesn't improve for 20 epochs, kill it.
    batch=16,             # You likely need to lower this further for Large
    workers=4,
    imgsz=640,
    dropout=0.1,         # CRITICAL: Keep this to fight the overfitting we saw in Medium
    augment=True,         # Keep this on
    project = 'Largest',
    name = 'yolo26X-pose-judo'
)