# Judo Annotation Suite (YOLOv11 / YOLO26)

A professional-grade Python GUI for creating and refining **Pose Estimation** and **Object Detection** datasets. This tool allows you to load raw videos, use pre-trained YOLO models to "auto-guess" annotations, and manually fine-tune keypoints and bounding boxes for high-accuracy Judo datasets.

## 🚀 Features

* **Dual Mode System:**
    * **Pose Mode:** Annotate 17-keypoint skeletons for Judo throws.
    * **Detect Mode:** Annotate standard bounding boxes for mats, referees, or equipment.
* **Smart Auto-Labeling:**
    * **Main Model:** Auto-annotate using your best model (e.g., `medbest.pt`).
    * **Comparison Mode:** Overlay predictions from a base model (e.g., `yolo26n`) to compare performance against your fine-tuned weights.
* **Cloud-Ready Workflow:** Fully compatible with Google Drive synchronization via `.env` configuration.
* **Manual Correction:**
    * **Drag Keypoints:** Fine-tune joint positions with pixel-perfect accuracy.
    * **Drag Bounding Boxes:** Manually resize boxes using corner handles.
    * **Visibility Toggling:** Right-click to cycle keypoint status (Visible 🟢, Occluded 🔴, Hidden ⚫).
* **YOLO Format:** Saves labels directly in the standard `.txt` format required for training.

## 🛠️ Installation

1.  **Clone or Download** this repository.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup Configuration:**
    Create a file named `.env` in the root directory (do not commit this file). Add your specific paths:
    ```ini
    # .env
    RAW_DATA_DIR=G:/My Drive/judo_datasetDONTDELETE   # Path to shared Drive folder
    PROCESSED_DATA_DIR=datasets/judo_pose             # Local path for training data
    MODEL_TRAIN_BASE=Models/yolo26x-pose.pt           # Base weights for training
    TRAIN_PROJECT_DIR=Largest                         # Training output folder name
    ```

## 🎮 Controls

| Action | Control |
| :--- | :--- |
| **Select Item** | Click any keypoint (Pose) or the box edge (Detect). |
| **Move Keypoint** | Left-Click & Drag a keypoint. |
| **Resize Box** | Select item, then Drag the **Yellow Corner Handles**. |
| **Toggle Visibility** | Right-Click a keypoint (Green 🟢 -> Red 🔴 -> Grey ⚫). |
| **Delete Item** | Select item and press `Del` or `Backspace`. |
| **Focus Mode** | Press `F` to dim background and focus on the selected person. |
| **Add Item** | Click `+ Add Person` (Pose) or `+ Add Object` (Detect). |

> **Visibility Legend (Pose Mode):**
> * 🟢 **Green:** Visible (Clear line of sight).
> * 🔴 **Red:** Occluded (Joint exists but is covered by Gi/body - **Crucial for Judo**).
> * ⚫ **Grey:** Hidden/Not labeled (Off-camera).

## 📝 Workflow Guide

### 1. Labeling (The Tool)
1.  Run the application:
    ```bash
    python main.py
    ```
2.  **Select Mode:** Choose **Pose Estimation** or **Object Detection** in the top left.
3.  **Import Video:** Load a raw `.mp4` file.
4.  **Load Model:**
    * Click **2a. Load Main** to use your fine-tuned model for auto-guessing.
    * (Optional) Click **2b. Load Base** to see how the default YOLO model performs.
5.  **Annotate & Save:** Correct the auto-guesses and click **Save Pair** (Green button).

### 2. Preparing for Training (The Bridge)
YOLO cannot train on the raw `judo_dataset` folder directly. You must split it into Train/Val sets and generate the configuration file.

1.  Run the splitter script:
    ```bash
    python datasplitter.py
    ```
2.  This script will:
    * Read your `.env` to find the source data.
    * Shuffle and split data (80% Train / 20% Val).
    * **Auto-generate** the `judo_pose.yaml` file with the correct absolute paths.
3.  Your data is now ready in `datasets/judo_pose`.

### 3. Training
Run the training script (which also reads from your `.env` configuration):
```bash
python train_test.py