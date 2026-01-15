import cv2

class VideoEngine:
    def __init__(self):
        self.cap = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.original_width = 0
        self.original_height = 0

    def load_video(self, path):
        """Initializes the video capture object."""
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError("Could not open video file")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_index = 0
        return self.total_frames

    def get_frame(self, index):
        """Retrieves a specific frame in RGB format."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR (OpenCV standard) to RGB (Qt standard)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def release(self):
        if self.cap:
            self.cap.release()