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
        self.focus_mode = False  # <--- NEW: Focus Mode Flag
        
        self.image_pixmap = None
        self.original_image_size = (0, 0)
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.annotations = [] 
        
        # Selection State
        self.selected_person_idx = -1
        self.selected_kpt_idx = -1
        
        # Dragging State
        self.dragging = False
        self.dragging_bbox = False
        self.bbox_handle_idx = -1 # 0:TL, 1:TR, 2:BR, 3:BL
        
        self.radius = 6
        self.handle_size = 8

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
        font = QFont("Arial", 8, QFont.Weight.Bold)
        painter.setFont(font)

        for p_idx, person in enumerate(self.annotations):
            kpts = person['keypoints']
            bbox = person.get('bbox', [0,0,0,0])
            
            is_selected = (p_idx == self.selected_person_idx)
            
            # --- FOCUS MODE LOGIC ---
            # If Focus Mode is ON and this person is NOT selected, make them a "Ghost"
            if self.focus_mode and self.selected_person_idx != -1 and not is_selected:
                opacity_factor = 40  # Very transparent (0-255)
                is_ghost = True
            else:
                opacity_factor = 255 # Opaque
                is_ghost = False

            # --- Draw Bounding Box ---
            self.draw_bbox(painter, bbox, is_selected, opacity_factor)

            # --- Draw Skeleton Lines ---
            line_color = QColor(0, 255, 255, 100)
            line_color.setAlpha(min(100, opacity_factor)) # Clamp alpha
            painter.setPen(QPen(line_color, 2)) 
            
            for i1, i2 in SKELETON_CONNECTIONS:
                p1 = self.norm_to_screen(kpts[i1][0], kpts[i1][1])
                p2 = self.norm_to_screen(kpts[i2][0], kpts[i2][1])
                painter.drawLine(p1, p2)

            # --- Draw Keypoints ---
            for k_idx, (nx, ny, vis) in enumerate(kpts):
                screen_pos = self.norm_to_screen(nx, ny)
                
                # Color Logic
                if vis == 2:   base_color = QColor(0, 255, 0)       # Green (Visible)
                elif vis == 1: base_color = QColor(255, 0, 0)       # Red (Occluded)
                else:          base_color = QColor(100, 100, 100)   # Grey (Hidden)
                
                base_color.setAlpha(opacity_factor)
                
                # Selection Highlight (Only if not a ghost)
                if is_selected and k_idx == self.selected_kpt_idx and not is_ghost:
                    painter.setBrush(QBrush(QColor(255, 255, 0))) # Yellow
                    r = self.radius + 3
                else:
                    painter.setBrush(QBrush(base_color))
                    r = self.radius
                
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(screen_pos, r, r)
                
                # Draw Numbers (Hide numbers for ghosts to reduce clutter)
                if self.show_numbers and not is_ghost:
                    painter.setPen(QColor(0, 0, 0)) 
                    text_rect = QRectF(screen_pos.x() - r, screen_pos.y() - r, r*2, r*2)
                    painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(k_idx))

    def draw_bbox(self, painter, bbox_norm, is_selected, alpha):
        cx, cy, w, h = bbox_norm
        x_tl = cx - w/2
        y_tl = cy - h/2
        
        pt1 = self.norm_to_screen(x_tl, y_tl)
        pt2 = self.norm_to_screen(x_tl + w, y_tl + h)
        rect = QRectF(pt1, pt2)
        
        if is_selected:
            color = QColor(255, 255, 0, 200) # Yellow
            style = Qt.PenStyle.SolidLine
            width = 2
        else:
            color = QColor(200, 200, 200, 80) # Faint grey
            style = Qt.PenStyle.DashLine
            width = 1

        # Apply Ghost transparency
        if alpha < 255:
            color.setAlpha(min(color.alpha(), alpha))

        painter.setPen(QPen(color, width, style))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        # Draw handles only if selected (never for ghosts)
        if is_selected and alpha == 255:
            handles = [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
            painter.setBrush(QBrush(QColor(255, 255, 0)))
            painter.setPen(Qt.PenStyle.NoPen)
            for pt in handles:
                painter.drawRect(QRectF(pt.x() - self.handle_size/2, pt.y() - self.handle_size/2, 
                                        self.handle_size, self.handle_size))

    def get_bbox_handles(self, bbox_norm):
        cx, cy, w, h = bbox_norm
        x_tl = cx - w/2
        y_tl = cy - h/2
        x_br = cx + w/2
        y_br = cy + h/2
        pt_tl = self.norm_to_screen(x_tl, y_tl)
        pt_tr = self.norm_to_screen(x_br, y_tl)
        pt_br = self.norm_to_screen(x_br, y_br)
        pt_bl = self.norm_to_screen(x_tl, y_br)
        return [pt_tl, pt_tr, pt_br, pt_bl]

    def mousePressEvent(self, event):
        if not self.annotations: return
        click_pos = event.position()
        
        # 1. Check Bounding Box Handles (Prioritize Selected Person)
        if self.selected_person_idx != -1:
            person = self.annotations[self.selected_person_idx]
            bbox = person.get('bbox', [0,0,0,0])
            handles = self.get_bbox_handles(bbox)
            for i, pt in enumerate(handles):
                if (pt - click_pos).manhattanLength() < 15:
                    self.dragging_bbox = True
                    self.bbox_handle_idx = i
                    self.dragging = True 
                    return

        # 2. Check Keypoints
        min_dist = 20 
        best_match = None
        
        for p_idx, person in enumerate(self.annotations):
            # Focus Mode Filter: If ON, ignore clicks on non-selected people (Ghosts)
            if self.focus_mode and self.selected_person_idx != -1 and p_idx != self.selected_person_idx:
                continue

            for k_idx, (nx, ny, vis) in enumerate(person['keypoints']):
                s_point = self.norm_to_screen(nx, ny)
                dist = (QPointF(s_point) - click_pos).manhattanLength()
                if dist < min_dist:
                    min_dist = dist
                    best_match = (p_idx, k_idx)

        if best_match:
            self.selected_person_idx, self.selected_kpt_idx = best_match
            self.dragging_bbox = False 
            
            if event.button() == Qt.MouseButton.RightButton:
                kp = self.annotations[self.selected_person_idx]['keypoints'][self.selected_kpt_idx]
                kp[2] = 1 if kp[2] == 2 else (0 if kp[2] == 1 else 2)
                self.update()
                
            elif event.button() == Qt.MouseButton.LeftButton:
                self.dragging = True
                self.update()
        else:
            # Only deselect if we aren't in Focus Mode
            # (In Focus Mode, clicking empty space usually means "I missed the dot", not "I want to exit focus")
            # But for standard UI behavior, let's allow deselection to exit focus naturally
            self.selected_person_idx = -1
            self.selected_kpt_idx = -1
            self.dragging_bbox = False
            self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging: return
        
        nx, ny = self.screen_to_norm(event.position().x(), event.position().y())

        if self.dragging_bbox and self.selected_person_idx != -1:
            person = self.annotations[self.selected_person_idx]
            cx, cy, w, h = person['bbox']
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            
            if self.bbox_handle_idx == 0: x1, y1 = nx, ny
            elif self.bbox_handle_idx == 1: x2, y1 = nx, ny
            elif self.bbox_handle_idx == 2: x2, y2 = nx, ny
            elif self.bbox_handle_idx == 3: x1, y2 = nx, ny
                
            final_x1, final_x2 = min(x1, x2), max(x1, x2)
            final_y1, final_y2 = min(y1, y2), max(y1, y2)
            
            new_w = final_x2 - final_x1
            new_h = final_y2 - final_y1
            new_cx = final_x1 + new_w/2
            new_cy = final_y1 + new_h/2
            
            person['bbox'] = [new_cx, new_cy, new_w, new_h]
            self.update()

        elif self.selected_person_idx != -1 and self.selected_kpt_idx != -1:
            kp = self.annotations[self.selected_person_idx]['keypoints'][self.selected_kpt_idx]
            kp[0] = nx
            kp[1] = ny
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.dragging_bbox = False
        self.bbox_handle_idx = -1
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
        return nx, ny