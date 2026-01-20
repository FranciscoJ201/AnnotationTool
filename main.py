import sys
import os
import shutil
import cv2
import torch 
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, 
                             QCheckBox, QMessageBox, QScrollArea, QFrame, QGroupBox)
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO

# Import our custom modules
from video_engine import VideoEngine
from annotator import AnnotationWidget, KEYPOINT_NAMES

class JudoAppQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Judo Pose Annotator v10 (Manual Add Person)")
        self.resize(1300, 800)

        # --- PROJECT DIRECTORY SETUP ---
        self.project_root = os.path.join(os.getcwd(), "judo_dataset")
        
        self.videos_storage_dir = os.path.join(self.project_root, "videos")
        os.makedirs(self.videos_storage_dir, exist_ok=True)

        self.current_video_name = ""
        self.current_images_dir = ""
        self.current_labels_dir = ""

        self.engine = VideoEngine()
        self.model = None 
        self.current_frame_img = None 
        self.current_video_path = ""
        self.is_playing = False
        
        self.timer = QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.next_frame_automatic)

        # --- MAIN LAYOUT ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # === LEFT PANEL ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.annotator = AnnotationWidget()
        left_layout.addWidget(self.annotator, stretch=1)

        # Controls
        play_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Prev")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self.next_frame)
        play_layout.addStretch()
        play_layout.addWidget(self.btn_prev)
        play_layout.addWidget(self.btn_play)
        play_layout.addWidget(self.btn_next)
        play_layout.addStretch()
        left_layout.addLayout(play_layout)

        file_layout = QHBoxLayout()
        self.btn_load = QPushButton("1. Import Video")
        self.btn_load.clicked.connect(self.load_video)
        self.btn_load_model = QPushButton("2. Load YOLO")
        self.btn_load_model.clicked.connect(self.load_yolo)
        self.chk_auto = QCheckBox("Auto-Guess New Frames")
        self.chk_auto.setChecked(True)
        self.btn_save = QPushButton("💾 Save Pair")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_pair)
        
        file_layout.addWidget(self.btn_load)
        file_layout.addWidget(self.btn_load_model)
        file_layout.addWidget(self.chk_auto)
        file_layout.addStretch()
        file_layout.addWidget(self.btn_save)
        left_layout.addLayout(file_layout)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_move)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        left_layout.addWidget(self.slider)

        self.lbl_status = QLabel(f"Project Root: {self.project_root}")
        left_layout.addWidget(self.lbl_status)
        
        main_layout.addWidget(left_panel, stretch=3)

        # === RIGHT PANEL ===
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.Shape.StyledPanel)
        right_panel.setStyleSheet("""
            QFrame { background-color: #f0f0f0; border-left: 1px solid #ccc; }
            QLabel { color: black; }
            QCheckBox { color: black; }
            QGroupBox { color: black; font-weight: bold; }
        """)
        right_layout = QVBoxLayout(right_panel)

        # NEW BUTTON: ADD PERSON
        self.btn_add_person = QPushButton("+ Add Person")
        self.btn_add_person.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        self.btn_add_person.clicked.connect(self.manual_add_person)
        right_layout.addWidget(self.btn_add_person)

        self.chk_show_nums = QCheckBox("Show Keypoint #")
        self.chk_show_nums.setStyleSheet("font-weight: bold; padding: 5px; color: black;")
        self.chk_show_nums.toggled.connect(self.toggle_numbers)
        right_layout.addWidget(self.chk_show_nums)
        
        legend_group = QGroupBox("Keypoint Legend")
        legend_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        for i, name in enumerate(KEYPOINT_NAMES):
            lbl = QLabel(f"<b>{i}</b> : {name}")
            lbl.setStyleSheet("font-family: monospace; font-size: 11px; color: black;")
            scroll_layout.addWidget(lbl)
            
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        legend_layout.addWidget(scroll)
        legend_group.setLayout(legend_layout)
        right_layout.addWidget(legend_group)
        
        instr = QLabel("Controls:\n- L-Click: Drag\n- R-Click: Toggle Vis\n(Green=Vis, Red=Occ)")
        instr.setStyleSheet("color: black; font-size: 10px; font-weight: bold;")
        right_layout.addWidget(instr)

        main_layout.addWidget(right_panel, stretch=1)
        self.slider_is_being_dragged = False

    def toggle_numbers(self, checked):
        self.annotator.show_numbers = checked
        self.annotator.update()

    # --- NEW: Manual Add Person Logic ---
    def manual_add_person(self):
        # Create a default "Standing Pose" in the center of the screen
        # Coordinates are Normalized (0.0 to 1.0)
        
        # Center X, Y
        cx, cy = 0.5, 0.5 
        
        # Offsets for a simple stick figure
        head_y = cy - 0.3
        shoulder_y = cy - 0.2
        hip_y = cy + 0.05
        knee_y = cy + 0.2
        ankle_y = cy + 0.35
        
        width = 0.05 # Half-width for arms/legs
        
        # 17 Keypoints (x, y, visibility=2)
        # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar
        # 5:LSh, 6:RSh, 7:LElb, 8:RElb, 9:LWri, 10:RWri
        # 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnk, 16:RAnk
        
        new_kpts = [
            [cx, head_y, 2],          # 0 Nose
            [cx+0.01, head_y-0.01, 2], # 1 LEye
            [cx-0.01, head_y-0.01, 2], # 2 REye
            [cx+0.02, head_y, 2],     # 3 LEar
            [cx-0.02, head_y, 2],     # 4 REar
            
            [cx+width, shoulder_y, 2], # 5 LShoulder
            [cx-width, shoulder_y, 2], # 6 RShoulder
            
            [cx+width+0.05, shoulder_y+0.1, 2], # 7 LElbow
            [cx-width-0.05, shoulder_y+0.1, 2], # 8 RElbow
            
            [cx+width+0.05, shoulder_y+0.2, 2], # 9 LWrist
            [cx-width-0.05, shoulder_y+0.2, 2], # 10 RWrist
            
            [cx+width, hip_y, 2],     # 11 LHip
            [cx-width, hip_y, 2],     # 12 RHip
            
            [cx+width, knee_y, 2],    # 13 LKnee
            [cx-width, knee_y, 2],    # 14 RKnee
            
            [cx+width, ankle_y, 2],   # 15 LAnkle
            [cx-width, ankle_y, 2],   # 16 RAnkle
        ]
        
        # Default Bbox (Center)
        new_person = {
            'bbox': [0.5, 0.5, 0.3, 0.8], 
            'keypoints': new_kpts
        }
        
        self.annotator.annotations.append(new_person)
        self.annotator.update()
        
        # Reset Save button color since we modified data
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Video", "", "Video (*.mp4 *.avi *.mov)")
        if path:
            filename = os.path.basename(path)
            destination_path = os.path.join(self.videos_storage_dir, filename)
            
            if os.path.abspath(path) != os.path.abspath(destination_path):
                try:
                    self.lbl_status.setText(f"Importing video...")
                    QApplication.processEvents()
                    shutil.copy(path, destination_path)
                except Exception as e:
                    QMessageBox.warning(self, "Import Error", f"Could not copy video: {e}")
                    return

            self.current_video_path = destination_path
            count = self.engine.load_video(self.current_video_path)
            
            self.current_video_name = os.path.splitext(filename)[0]
            video_dataset_root = os.path.join(self.project_root, self.current_video_name)
            self.current_images_dir = os.path.join(video_dataset_root, "images")
            self.current_labels_dir = os.path.join(video_dataset_root, "labels")
            
            os.makedirs(self.current_images_dir, exist_ok=True)
            os.makedirs(self.current_labels_dir, exist_ok=True)
            
            self.slider.setRange(0, count - 1)
            self.slider.setValue(0)
            self.seek_frame(0)
            self.lbl_status.setText(f"Loaded: {filename} | Data Folder: /{self.current_video_name}")

    def load_yolo(self):
        ENGINE_PATH = 'yolo11x-pose.engine'
        try:
            self.lbl_status.setText("Checking YOLO model status...")
            QApplication.processEvents()

            if not os.path.exists(ENGINE_PATH):
                if not torch.cuda.is_available():
                    print('Swapping to Cpu, NO GPU detected')
                    self.lbl_status.setText("No GPU detected. Falling back to CPU (Nano model)...")
                    QApplication.processEvents()
                    self.model = YOLO('yolo11n-pose.pt')
                else:
                    print(f"Exporting engine...")
                    self.lbl_status.setText(f"GPU Detected! Exporting TensorRT Engine (Please Wait)...")
                    QApplication.processEvents()
                    model = YOLO('yolo11x-pose.pt')
                    model.export(format='engine', half=True)
                    self.lbl_status.setText("Export Complete! Loading Engine...")
                    QApplication.processEvents()
                    self.model = YOLO(ENGINE_PATH)
            else:
                self.lbl_status.setText(f"Loading cached Engine: {ENGINE_PATH}")
                QApplication.processEvents()
                self.model = YOLO(ENGINE_PATH)

            self.lbl_status.setText("YOLO Model Loaded!")
            self.btn_load_model.setStyleSheet("background-color: #d4edda")

        except Exception as e:
            QMessageBox.critical(self, "Model Error", f"Could not load/export YOLO: {e}")
            self.lbl_status.setText("Error loading model.")

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("|| Pause")
            self.timer.start()
        else:
            self.btn_play.setText("▶ Play")
            self.timer.stop()

    def next_frame_automatic(self):
        if self.engine.current_frame_index < self.engine.total_frames - 1:
            next_idx = self.engine.current_frame_index + 1
            self.slider.blockSignals(True)
            self.slider.setValue(next_idx)
            self.slider.blockSignals(False)
            self.seek_frame(next_idx)
        else:
            self.toggle_play()

    def next_frame(self):
        self.stop_playback()
        if self.engine.current_frame_index < self.engine.total_frames - 1:
            self.slider.setValue(self.engine.current_frame_index + 1)

    def prev_frame(self):
        self.stop_playback()
        if self.engine.current_frame_index > 0:
            self.slider.setValue(self.engine.current_frame_index - 1)

    def stop_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.setText("▶ Play")
            self.timer.stop()

    def on_slider_move(self, value):
        self.seek_frame(value)

    def slider_pressed(self):
        self.slider_is_being_dragged = True
        self.timer.stop()

    def slider_released(self):
        self.slider_is_being_dragged = False
        if self.is_playing:
            self.timer.start()

    def seek_frame(self, idx):
        self.engine.current_frame_index = idx
        img = self.engine.get_frame(idx)
        if img is not None:
            self.current_frame_img = img
            self.annotator.set_image(img)
            
            if self.try_load_existing_labels(idx):
                self.lbl_status.setText(f"Frame {idx}: Loaded Saved Labels ✅")
                self.btn_save.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
            elif self.chk_auto.isChecked() and self.model:
                self.run_inference(img)
                self.lbl_status.setText(f"Frame {idx}: Auto-Guessed 🤖")
                self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            else:
                self.annotator.annotations = [] 
                self.annotator.update()
                self.lbl_status.setText(f"Frame {idx}: No Data")
                self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

    def try_load_existing_labels(self, idx):
        if not self.current_labels_dir: return False
        filename = f"{self.current_video_name}_{idx:06d}.txt"
        path = os.path.join(self.current_labels_dir, filename)
        if not os.path.exists(path): return False
        
        new_annotations = []
        try:
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    bbox = parts[1:5]
                    raw_kpts = parts[5:]
                    formatted_kpts = []
                    for i in range(0, len(raw_kpts), 3):
                        x = raw_kpts[i]
                        y = raw_kpts[i+1]
                        v = int(raw_kpts[i+2])
                        formatted_kpts.append([x, y, v])
                    new_annotations.append({'bbox': bbox, 'keypoints': formatted_kpts})
            self.annotator.annotations = new_annotations
            self.annotator.update()
            return True
        except Exception:
            return False

    def run_inference(self, img):
        if not self.model: return
        results = self.model(img, verbose=False)
        self.annotator.annotations = []
        if results and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.xyn.cpu().numpy()
            boxes = results[0].boxes.xywhn.cpu().numpy() 
            for i, kpts in enumerate(keypoints_data):
                formatted_kpts = []
                for kp in kpts:
                    x, y = kp
                    vis = 0 if (x==0 and y==0) else 2 
                    formatted_kpts.append([float(x), float(y), vis])
                bbox = boxes[i].tolist() if i < len(boxes) else [0,0,0,0]
                self.annotator.annotations.append({'bbox': bbox, 'keypoints': formatted_kpts})
        self.annotator.update()

    def save_pair(self):
        if not self.current_images_dir or not self.current_labels_dir: 
            QMessageBox.warning(self, "Error", "No video folder active.")
            return
        if self.current_frame_img is None: return

        base_filename = f"{self.current_video_name}_{self.engine.current_frame_index:06d}"
        
        txt_path = os.path.join(self.current_labels_dir, f"{base_filename}.txt")
        try:
            with open(txt_path, "w") as f:
                for person in self.annotator.annotations:
                    line = [0] 
                    line.extend(person['bbox']) 
                    for k in person['keypoints']:
                        line.extend(k)
                    f.write(" ".join(map(str, line)) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Txt Error: {e}")
            return

        img_path = os.path.join(self.current_images_dir, f"{base_filename}.jpg")
        try:
            bgr_img = cv2.cvtColor(self.current_frame_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, bgr_img)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Image Error: {e}")
            return
            
        self.lbl_status.setText(f"Saved to /{self.current_video_name}: {base_filename}")
        self.btn_save.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JudoAppQt()
    window.show()
    sys.exit(app.exec())