from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont

# --- COCO SKELETON CONFIG ---
KEYPOINT_NAMES = [
    "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear", 
    "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow", "L-Wrist", "R-Wrist",
    "L-Hip", "R-Hip", "L-Knee", "R-Knee", "L-Ankle", "R-Ankle"
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12),                         # Torso
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]

class AnnotationWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #222;")
        
        self.show_numbers = False 
        
        self.image_pixmap = None
        self.original_image_size = (0, 0)
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.annotations = [] 
        
        self.selected_person_idx = -1
        self.selected_kpt_idx = -1
        self.dragging = False
        self.radius = 6

    def set_image(self, numpy_img):
        h, w, ch = numpy_img.shape
        self.original_image_size = (w, h)
        bytes_per_line = ch * w
        q_img = QImage(numpy_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_pixmap = QPixmap.fromImage(q_img)
        self.update_display_geometry()
        self.update()

    def update_display_geometry(self):
        if not self.image_pixmap: return
        w_widget = self.width()
        h_widget = self.height()
        w_img = self.original_image_size[0]
        h_img = self.original_image_size[1]
        
        if w_img > 0 and h_img > 0:
            scale_w = w_widget / w_img
            scale_h = h_widget / h_img
            self.scale_factor = min(scale_w, scale_h)
            display_w = w_img * self.scale_factor
            display_h = h_img * self.scale_factor
            self.offset_x = (w_widget - display_w) / 2
            self.offset_y = (h_widget - display_h) / 2

    def resizeEvent(self, event):
        self.update_display_geometry()
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        
        # 1. Draw Image
        if self.image_pixmap:
            dest_w = int(self.original_image_size[0] * self.scale_factor)
            dest_h = int(self.original_image_size[1] * self.scale_factor)
            painter.drawPixmap(int(self.offset_x), int(self.offset_y), dest_w, dest_h, self.image_pixmap)

        # 2. Draw Annotations
        # Use a slightly smaller font (8pt) so it fits inside the dots
        if self.show_numbers:
            font = QFont("Arial", 8, QFont.Weight.Bold)
            painter.setFont(font)

        for p_idx, person in enumerate(self.annotations):
            kpts = person['keypoints']
            
            # Draw Skeleton Lines
            painter.setPen(QPen(QColor(0, 255, 255, 100), 2)) 
            for i1, i2 in SKELETON_CONNECTIONS:
                p1 = self.norm_to_screen(kpts[i1][0], kpts[i1][1])
                p2 = self.norm_to_screen(kpts[i2][0], kpts[i2][1])
                painter.drawLine(p1, p2)

            # Draw Keypoints
            for k_idx, (nx, ny, vis) in enumerate(kpts):
                screen_pos = self.norm_to_screen(nx, ny)
                
                # Color Logic
                if vis == 2:   color = QColor(0, 255, 0)       # Green (Visible)
                elif vis == 1: color = QColor(255, 0, 0)       # Red (Occluded)
                else:          color = QColor(100, 100, 100)   # Grey (Hidden)
                
                # Selection Highlight
                if p_idx == self.selected_person_idx and k_idx == self.selected_kpt_idx:
                    painter.setBrush(QBrush(QColor(255, 255, 0))) # Yellow
                    r = self.radius + 3
                else:
                    painter.setBrush(QBrush(color))
                    r = self.radius
                
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(screen_pos, r, r)
                
                # --- NEW CENTERED TEXT LOGIC ---
                if self.show_numbers:
                    # Black text usually looks best on colored dots
                    painter.setPen(QColor(0, 0, 0)) 
                    
                    # Create a rectangle exactly covering the dot
                    # We use (2*r) because r is radius, we need diameter
                    text_rect = QRectF(screen_pos.x() - r, screen_pos.y() - r, r*2, r*2)
                    
                    # Draw text aligned to the center of that rectangle
                    painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(k_idx))

    def mousePressEvent(self, event):
        if not self.annotations: return
        
        click_pos = event.position()
        min_dist = 20 
        best_match = None
        
        for p_idx, person in enumerate(self.annotations):
            for k_idx, (nx, ny, vis) in enumerate(person['keypoints']):
                s_point = self.norm_to_screen(nx, ny)
                dist = (QPointF(s_point) - click_pos).manhattanLength()
                if dist < min_dist:
                    min_dist = dist
                    best_match = (p_idx, k_idx)

        if best_match:
            self.selected_person_idx, self.selected_kpt_idx = best_match
            
            if event.button() == Qt.MouseButton.RightButton:
                kp = self.annotations[self.selected_person_idx]['keypoints'][self.selected_kpt_idx]
                kp[2] = 1 if kp[2] == 2 else (0 if kp[2] == 1 else 2)
                
                # Clear selection immediately
                self.selected_person_idx = -1
                self.selected_kpt_idx = -1
                self.update()
                
            elif event.button() == Qt.MouseButton.LeftButton:
                self.dragging = True
                self.update()
        else:
            self.selected_person_idx = -1
            self.selected_kpt_idx = -1
            self.update()

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_person_idx != -1:
            nx, ny = self.screen_to_norm(event.position().x(), event.position().y())
            kp = self.annotations[self.selected_person_idx]['keypoints'][self.selected_kpt_idx]
            kp[0] = nx
            kp[1] = ny
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        # Clear selection on release
        self.selected_person_idx = -1
        self.selected_kpt_idx = -1
        self.update()

    def norm_to_screen(self, nx, ny):
        img_w, img_h = self.original_image_size
        sx = (nx * img_w * self.scale_factor) + self.offset_x
        sy = (ny * img_h * self.scale_factor) + self.offset_y
        return QPointF(sx, sy)

    def screen_to_norm(self, sx, sy):
        img_w, img_h = self.original_image_size
        nx = (sx - self.offset_x) / self.scale_factor / img_w
        ny = (sy - self.offset_y) / self.scale_factor / img_h
        return max(0, min(1, nx)), max(0, min(1, ny))