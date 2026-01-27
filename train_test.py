from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load the model
    # It will download yolo26n-pose.pt automatically if you don't have it
    model = YOLO('yolo26n-pose.pt')

    # 2. Train the model
    results = model.train(
        data='judo_pose.yaml',   # Path to your config file
        epochs=200,               # Quick test
        imgsz=640,               # Standard image size
        project='runs',          # Folder to save results
        name='dry_run',           # Folder name for this specific run
        workers=6,
        batch=64
    )