import sys
import os
import shutil
import cv2
import torch 
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, QFileDialog, 
                             QCheckBox, QMessageBox, QScrollArea, QFrame, QGroupBox,
                             QRadioButton, QButtonGroup, QInputDialog)
from PyQt6.QtCore import Qt, QTimer
from ultralytics import YOLO

# Import our custom modules
from video_engine import VideoEngine
from annotator import AnnotationWidget, KEYPOINT_NAMES

class JudoAppQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Judo Annotation Suite")
        self.resize(1300, 800)

        # --- STATE MANAGEMENT ---
        # "pose" or "detect"
        self.app_mode = "pose" 

        # --- PROJECT DIRECTORY SETUP ---
        self.project_root = os.path.join(os.getcwd(), "judo_datasetDONTDELETE")
        self.videos_storage_dir = os.path.join(self.project_root, "videos")
        
        # Define the global classes file path
        self.classes_file_path = os.path.join(self.project_root, "classes.txt")
        
        os.makedirs(self.videos_storage_dir, exist_ok=True)

        self.current_video_name = ""
        self.current_video_path = ""
        
        # These update dynamically based on mode
        self.active_images_dir = ""
        self.active_labels_dir = ""

        self.engine = VideoEngine()
        self.model = None 
        self.current_frame_img = None 
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
        
        # --- MODE SWITCHER ---
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.rb_pose = QRadioButton("Mode: Pose Estimation")
        self.rb_detect = QRadioButton("Mode: Object Detection")
        self.rb_pose.setChecked(True)
        self.mode_group.addButton(self.rb_pose)
        self.mode_group.addButton(self.rb_detect)
        
        self.rb_pose.toggled.connect(self.on_mode_change)
        
        mode_layout.addWidget(QLabel("<b>MODE:</b>"))
        mode_layout.addWidget(self.rb_pose)
        mode_layout.addWidget(self.rb_detect)
        mode_layout.addStretch()
        left_layout.addLayout(mode_layout)

        # Annotator
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

        # --- FILE / MODEL CONTROLS ---
        file_layout = QHBoxLayout()
        self.btn_load = QPushButton("1. Import Video")
        self.btn_load.clicked.connect(self.load_video)
        
        self.btn_load_model = QPushButton("2a. Load Main Model") # Renamed slightly
        self.btn_load_model.clicked.connect(self.load_yolo_main)

        # --- NEW COMPARISON BUTTON ---
        self.btn_load_compare = QPushButton("2b. Load Base Model to Compare (26n)")
        self.btn_load_compare.setStyleSheet("background-color: #e0f7fa; color: black;")
        self.btn_load_compare.clicked.connect(self.load_yolo_compare)
        # -----------------------------

        self.chk_auto = QCheckBox("Auto-Guess")
        self.chk_auto.setChecked(True)
        self.btn_save = QPushButton("💾 Save Pair")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_pair)
        
        file_layout.addWidget(self.btn_load)
        file_layout.addWidget(self.btn_load_model)
        file_layout.addWidget(self.btn_load_compare) # Add to layout
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

        # DYNAMIC BUTTON (Add Person / Add Object)
        self.btn_add_item = QPushButton("+ Add Person")
        self.btn_add_item.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 5px;")
        self.btn_add_item.clicked.connect(self.manual_add_item)
        right_layout.addWidget(self.btn_add_item)

        # DELETE BUTTON
        self.btn_del_item = QPushButton("🗑 Delete Selected")
        self.btn_del_item.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 5px;")
        self.btn_del_item.clicked.connect(self.delete_selected_item)
        right_layout.addWidget(self.btn_del_item)
        
        # Focus Mode
        self.chk_focus = QCheckBox("Focus Selected (F)")
        self.chk_focus.setStyleSheet("font-weight: bold; padding: 5px; color: black;")
        self.chk_focus.toggled.connect(self.toggle_focus)
        right_layout.addWidget(self.chk_focus)

        self.chk_show_nums = QCheckBox("Show Keypoint #")
        self.chk_show_nums.setStyleSheet("font-weight: bold; padding: 5px; color: black;")
        self.chk_show_nums.toggled.connect(self.toggle_numbers)
        right_layout.addWidget(self.chk_show_nums)
        
        # Legend (Only relevant for Pose)
        self.legend_group = QGroupBox("Keypoint Legend")
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
        self.legend_group.setLayout(legend_layout)
        right_layout.addWidget(self.legend_group)
        
        instr = QLabel("Controls:\n- L-Click: Drag\n- R-Click: Toggle Vis\n- Del: Delete Item")
        instr.setStyleSheet("color: black; font-size: 10px; font-weight: bold;")
        right_layout.addWidget(instr)

        main_layout.addWidget(right_panel, stretch=1)
        self.slider_is_being_dragged = False

        # Init directories
        self.update_directories()
        
        # Ensure "person" is class 0 if file doesn't exist
        if not os.path.exists(self.classes_file_path):
            with open(self.classes_file_path, 'w') as f:
                f.write("person\n")

    # --- CLASSES MANAGEMENT ---
    def get_class_id(self, label_name):
        existing_classes = []
        if os.path.exists(self.classes_file_path):
            with open(self.classes_file_path, 'r') as f:
                existing_classes = [line.strip() for line in f.readlines() if line.strip()]
        
        if label_name in existing_classes:
            return existing_classes.index(label_name)
        else:
            with open(self.classes_file_path, 'a') as f:
                f.write(f"{label_name}\n")
            return len(existing_classes)

    def get_class_name(self, class_id):
        if os.path.exists(self.classes_file_path):
            with open(self.classes_file_path, 'r') as f:
                existing_classes = [line.strip() for line in f.readlines() if line.strip()]
            if 0 <= class_id < len(existing_classes):
                return existing_classes[class_id]
        return f"Obj {class_id}"

    # --- MODE & UI LOGIC ---
    def on_mode_change(self):
        if self.rb_pose.isChecked():
            self.app_mode = "pose"
            self.btn_add_item.setText("+ Add Person")
            self.btn_load_model.setText("2a. Load Main (Pose)")
            self.legend_group.show()
            self.chk_show_nums.show()
            self.btn_load_compare.show() # Show compare button in pose mode
        else:
            self.app_mode = "detect"
            self.btn_add_item.setText("+ Add Object")
            self.btn_load_model.setText("2a. Load Main (Detect)")
            self.legend_group.hide()
            self.chk_show_nums.hide()
            self.btn_load_compare.hide() # Hide compare button in detect mode
        
        self.model = None
        self.btn_load_model.setStyleSheet("") 
        self.btn_load_compare.setStyleSheet("background-color: #e0f7fa; color: black;")
        self.update_directories()
        self.seek_frame(self.engine.current_frame_index)

    def update_directories(self):
        if not self.current_video_name:
            return
        video_root = os.path.join(self.project_root, self.current_video_name)
        
        if self.app_mode == "pose":
            mode_root = os.path.join(video_root, "pose")
        else:
            mode_root = os.path.join(video_root, "detect")
            
        self.active_images_dir = os.path.join(mode_root, "images")
        self.active_labels_dir = os.path.join(mode_root, "labels")
        
        os.makedirs(self.active_images_dir, exist_ok=True)
        os.makedirs(self.active_labels_dir, exist_ok=True)
        
        self.lbl_status.setText(f"Active Mode: {self.app_mode.upper()} | Folder: .../{self.current_video_name}/{self.app_mode}/")

    def toggle_numbers(self, checked):
        self.annotator.show_numbers = checked
        self.annotator.update()

    def toggle_focus(self, checked):
        self.annotator.focus_mode = checked
        self.annotator.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F:
            self.chk_focus.setChecked(not self.chk_focus.isChecked())
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected_item()
        else:
            super().keyPressEvent(event)

    def delete_selected_item(self):
        idx = self.annotator.selected_idx
        if idx != -1 and idx < len(self.annotator.annotations):
            del self.annotator.annotations[idx]
            self.annotator.selected_idx = -1
            self.annotator.selected_kpt_idx = -1
            self.annotator.update()
            self.lbl_status.setText(f"Deleted item at index {idx}.")
            self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        else:
            QMessageBox.information(self, "Delete", "No item selected. Click a person/object first to select them.")

    def manual_add_item(self):
        cx, cy = 0.5, 0.5 
        
        if self.app_mode == "pose":
            class_id = self.get_class_id("person")
            head_y = cy - 0.3
            shoulder_y = cy - 0.2
            hip_y = cy + 0.05
            knee_y = cy + 0.2
            ankle_y = cy + 0.35
            width = 0.05 
            
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
            
            new_item = {
                'type': 'person', 'class_id': class_id,
                'bbox': [0.5, 0.5, 0.3, 0.8], 'keypoints': new_kpts
            }
        else:
            text, ok = QInputDialog.getText(self, "Add Object", "Enter Label Name (e.g. chair, ball):")
            if not ok or not text: return
            class_id = self.get_class_id(text.lower().strip())
            new_item = {
                'type': 'object', 'label': text.lower().strip(), 'class_id': class_id,
                'bbox': [0.5, 0.5, 0.2, 0.2], 'keypoints': None
            }

        self.annotator.annotations.append(new_item)
        self.annotator.update()
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
            self.update_directories()
            self.slider.setRange(0, count - 1)
            self.slider.setValue(0)
            self.seek_frame(0)

    # --- MODEL LOADING LOGIC ---
    def load_yolo_main(self):
        """Loads the default main model depending on current mode."""
        if self.app_mode == "pose":
            engine = 'Models/medbest.engine'
            pt = 'Models/medbest.pt'
        else:
            engine = 'Models/yolo11x.engine'
            pt = 'Models/yolo26n.pt'
        
        self._load_model_generic(engine, pt)

    def load_yolo_compare(self):
        """Forces app to Pose mode and loads the comparison model."""
        # Force Pose Mode if not active
        if not self.rb_pose.isChecked():
            self.rb_pose.setChecked(True)
            self.on_mode_change()

        # Define paths for the comparison model
        # Assuming you name it yolo26n-pose.pt / .engine
        engine = 'Models/yolo26n-pose.engine'
        pt = 'Models/yolo26n-pose.pt'

        self._load_model_generic(engine, pt)

    def _load_model_generic(self, engine_path, pt_path):
        """Reusable helper to load any YOLO model."""
        self.lbl_status.setText(f"Loading Model: {pt_path} ...")
        QApplication.processEvents()

        model_loaded = False
        
        # 1. Try GPU (TensorRT Engine)
        if torch.cuda.is_available():
            try:
                if os.path.exists(engine_path):
                    self.lbl_status.setText(f"Loading Engine: {engine_path}")
                    QApplication.processEvents()
                    self.model = YOLO(engine_path)
                    model_loaded = True
                else:
                    print(f"GPU Detected! Checking export capability...")
                    self.lbl_status.setText(f"Exporting to Engine ({engine_path})...")
                    QApplication.processEvents()
                    
                    if os.path.exists(pt_path):
                        model = YOLO(pt_path) 
                        model.export(format='engine', half=True)
                        self.lbl_status.setText("Export Complete! Loading...")
                        QApplication.processEvents()
                        self.model = YOLO(engine_path)
                        model_loaded = True
                    else:
                        print(f"Missing source PT file: {pt_path}")
            except Exception as e:
                print(f"Warning: GPU acceleration/export failed. Error: {e}")
                self.lbl_status.setText(f"GPU Error. Switching to CPU...")
                QApplication.processEvents()
                model_loaded = False 
        
        # 2. Try CPU (PT File)
        if not model_loaded:
            try:
                if os.path.exists(pt_path):
                    print(f'Loading CPU model ({pt_path})...')
                    self.lbl_status.setText(f"Loading CPU Model ({pt_path})...")
                    QApplication.processEvents()
                    self.model = YOLO(pt_path)
                    model_loaded = True
                else:
                    QMessageBox.critical(self, "Model Error", f"Could not find model file:\n{pt_path}")
                    self.lbl_status.setText("Error loading model.")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Model Error", f"Critical: Could not load CPU model: {e}")
                self.lbl_status.setText("Error loading model.")
                return

        if model_loaded:
            self.lbl_status.setText(f"Loaded: {self.model.model_name}")
            print(f'Loaded Model: {self.model.model_name}')
            
            # Visual feedback: Turn active button green, others neutral
            # Note: This simple check assumes specific button usage
            if pt_path == 'Models/best.pt' or pt_path == 'Models/yolo26n.pt':
                 self.btn_load_model.setStyleSheet("background-color: #d4edda")
                 self.btn_load_compare.setStyleSheet("background-color: #e0f7fa")
            else:
                 self.btn_load_compare.setStyleSheet("background-color: #d4edda")
                 self.btn_load_model.setStyleSheet("")

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
            self.annotator.selected_idx = -1
            self.annotator.selected_kpt_idx = -1
            
            if self.try_load_existing_labels(idx):
                self.lbl_status.setText(f"Frame {idx}: Loaded Saved ({self.app_mode}) ✅")
                self.btn_save.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
            elif self.chk_auto.isChecked() and self.model:
                self.run_inference(img)
                self.lbl_status.setText(f"Frame {idx}: Auto-Guessed ({self.app_mode}) 🤖")
                self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            else:
                self.annotator.annotations = [] 
                self.annotator.update()
                self.lbl_status.setText(f"Frame {idx}: No Data")
                self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

    def try_load_existing_labels(self, idx):
        if not self.active_labels_dir: return False
        filename = f"{self.current_video_name}_{idx:06d}.txt"
        path = os.path.join(self.active_labels_dir, filename)
        if not os.path.exists(path): return False
        
        new_annotations = []
        try:
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    class_id = int(parts[0])
                    bbox = parts[1:5]
                    
                    if len(parts) > 5:
                        if self.app_mode != "pose": continue 
                        raw_kpts = parts[5:]
                        formatted_kpts = []
                        for i in range(0, len(raw_kpts), 3):
                            x = raw_kpts[i]
                            y = raw_kpts[i+1]
                            v = int(raw_kpts[i+2])
                            formatted_kpts.append([x, y, v])
                        new_annotations.append({
                            'type': 'person', 'class_id': class_id, 
                            'bbox': bbox, 'keypoints': formatted_kpts
                        })
                    else:
                        if self.app_mode != "detect": continue
                        label_name = self.get_class_name(class_id)
                        new_annotations.append({
                            'type': 'object', 'label': label_name, 'class_id': class_id,
                            'bbox': bbox, 'keypoints': None
                        })
            
            if new_annotations:
                self.annotator.annotations = new_annotations
                self.annotator.update()
                return True
            return False
            
        except Exception:
            return False

    def run_inference(self, img):
        if not self.model: return
        results = self.model(img, verbose=False)
        self.annotator.annotations = []
        
        if not results: return

        if self.app_mode == "pose" and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.xyn.cpu().numpy()
            boxes = results[0].boxes.xywhn.cpu().numpy() 
            for i, kpts in enumerate(keypoints_data):
                formatted_kpts = []
                for kp in kpts:
                    x, y = kp
                    vis = 0 if (x==0 and y==0) else 2 
                    formatted_kpts.append([float(x), float(y), vis])
                bbox = boxes[i].tolist() if i < len(boxes) else [0,0,0,0]
                self.annotator.annotations.append({
                    'type': 'person', 'class_id': 0,
                    'bbox': bbox, 'keypoints': formatted_kpts
                })
        
        elif self.app_mode == "detect" and results[0].boxes is not None:
            boxes = results[0].boxes.xywhn.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            for i, box in enumerate(boxes):
                cls = int(classes[i])
                label_name = results[0].names[cls]
                self.annotator.annotations.append({
                    'type': 'object', 'label': label_name, 'class_id': cls,
                    'bbox': box.tolist(), 'keypoints': None
                })
                
        self.annotator.update()

    def save_pair(self):
        if not self.active_images_dir or not self.active_labels_dir: 
            QMessageBox.warning(self, "Error", "No valid folder for current mode.")
            return
        if self.current_frame_img is None: return

        base_filename = f"{self.current_video_name}_{self.engine.current_frame_index:06d}"
        txt_path = os.path.join(self.active_labels_dir, f"{base_filename}.txt")
        
        try:
            with open(txt_path, "w") as f:
                for item in self.annotator.annotations:
                    line = [item['class_id']]
                    line.extend(item['bbox'])
                    if item.get('keypoints'):
                        for k in item['keypoints']:
                            line.extend(k)
                    
                    f.write(" ".join(map(str, line)) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Txt Error: {e}")
            return

        img_path = os.path.join(self.active_images_dir, f"{base_filename}.jpg")
        try:
            bgr_img = cv2.cvtColor(self.current_frame_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, bgr_img)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Image Error: {e}")
            return
            
        self.lbl_status.setText(f"Saved to .../{self.app_mode}/: {base_filename}")
        self.btn_save.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JudoAppQt()
    window.show()
    sys.exit(app.exec())