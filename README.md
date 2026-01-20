# Judo Pose Annotation Tool (YOLOv11)

A custom Python GUI for creating and refining Pose Estimation datasets. This tool allows you to load raw videos, use a pre-trained YOLO model to "auto-guess" poses, and manually fine-tune keypoints and bounding boxes for high-accuracy Judo datasets.

## 🚀 Features

* **Video Navigation:** Frame-by-frame stepping or slider seeking.
* **Auto-Labeling:** Integration with YOLOv11 to generate initial keypoint guesses.
* **Manual Correction:**
    * **Drag Keypoints:** Fine-tune joint positions.
    * **Drag Bounding Boxes:** Manually resize boxes using corner handles.
    * **Add Person:** Insert a default skeleton for missed detections.
* **Visibility Toggling:** Right-click to cycle keypoint status (Visible 🟢, Occluded 🔴, Hidden ⚫).
* **YOLO Format:** Saves labels directly in the standard `.txt` format required for training.

## 🛠️ Installation

1.  **Clone or Download** this repository.
2.  **Install Dependencies:**
    Ensure you have Python 3.8+ installed.

    ```bash
    pip install opencv-python PyQt6 ultralytics torch numpy
    ```

3.  **Project Structure:**
    Keep your directory organized like this:

    ```text
    /project_root
    ├── main.py              # The App Entry Point
    ├── annotator.py         # Custom Widget for drawing/dragging
    ├── video_engine.py      # Video loading logic
    ├── split_dataset.py     # Utility to prepare data for training
    ├── judo_dataset/        # (Auto-created) Stores raw images/labels
    └── datasets/            # (Auto-created) Final training ready data
    ```

## 🎮 Controls

| Action | Control |
| :--- | :--- |
| **Select Person** | Click any keypoint on a person. |
| **Move Keypoint** | Left-Click & Drag a keypoint. |
| **Resize Bounding Box** | Select person, then Drag the **Yellow Corner Handles**. |
| **Toggle Visibility** | Right-Click a keypoint (Green -> Red -> Grey). |
| **Navigation** | Use On-screen buttons or Slider. |
| **Add Person** | Click `+ Add Person` button (Adds stick figure to center). |

> **Visibility Legend:**
> * 🟢 **Green:** Visible (Clear line of sight).
> * 🔴 **Red:** Occluded (Joint exists but is covered by cloth/body).
> * ⚫ **Grey:** Hidden/Not labeled.

## 📝 Workflow Guide

### 1. Labeling (The Tool)
1.  Run the application: `python main.py`
2.  **Import Video:** Click "Import Video" to load a raw `.mp4` file.
3.  **Load YOLO:** Click "Load YOLO" to enable auto-guessing (uses `yolo11x-pose` or CPU fallback).
4.  **Annotate:**
    * If the model misses, click `+ Add Person`.
    * If the box is too loose, drag the yellow corners to tighten it.
    * Right-click hidden joints to mark them as "Occluded" (Red).
5.  **Save:** Click "Save Pair". This saves the image to `images/` and the coordinates to `labels/`.

### 2. Preparing for Training (The Bridge)
YOLO cannot train on the raw `judo_dataset` folder directly. You must split it into Train/Val sets.

1.  Open `split_dataset.py`.
2.  Run it:
    ```bash
    python split_dataset.py
    ```
3.  This creates a clean folder at `datasets/judo_pose` with shuffled `train` and `val` subfolders.

### 3. Training the Model
1.  Create a config file named `judo_pose.yaml` in your root directory:
    ```yaml
    path: datasets/judo_pose
    train: train/images
    val: val/images
    kpt_shape: [17, 3]
    flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    names:
      0: person
    ```
2.  Run the training command (ensure you are inside the project root):
    ```bash
    yolo pose train data=judo_pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
    ```

## ⚠️ Important Notes
* **Bounding Boxes:** The tool saves the bounding box *exactly* as it appears on screen. If you move a hand outside the box, **you must resize the box manually** using the yellow handles, or the training data will be invalid.
* **Cumulative Training:** When adding new data, always run `split_dataset.py` again and retrain on the **full** dataset (old + new images) to prevent the model from forgetting previous knowledge.