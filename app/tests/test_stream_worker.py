"""
Minimal test application for StreamWorker
Use this to verify Phase 1 implementation works correctly
"""
import sys
import numpy as np
import cv2 as cv
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage

# Mock imports for testing
sys.path.append('app')
from models.camera import Camera
from utils.stream_worker import StreamWorker, StreamState

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("StreamWorker Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Waiting for connection...")
        layout.addWidget(self.video_label)
        
        # Status label
        self.status_label = QLabel("Status: Initializing...")
        layout.addWidget(self.status_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_button)
        
        self.roi_button = QPushButton("Edit ROI Mode")
        self.roi_button.clicked.connect(self.toggle_roi_mode)
        button_layout.addWidget(self.roi_button)
        
        self.reconnect_button = QPushButton("Force Reconnect")
        self.reconnect_button.clicked.connect(self.force_reconnect)
        button_layout.addWidget(self.reconnect_button)
        
        layout.addLayout(button_layout)
        
        # Initialize camera and worker
        self.init_camera()
        
    def init_camera(self):
        """Initialize test camera and stream worker"""
        # Create test camera (webcam)
        self.camera = Camera(
            name="Test Camera",
            ip_address="192.168.88.235",
            port=554,
            username="admin123",
            password="admin123",
            stream_path="stream1",
        )
        
        # Create stream worker
        self.worker = StreamWorker(
            self.camera,
            bg_params={},  # Empty for Phase 1
            contour_params={}  # Empty for Phase 1
        )
        
        # Connect signals
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.state_changed.connect(self.on_state_changed)
        self.worker.connection_changed.connect(self.on_connection_changed)
        self.worker.error_occurred.connect(self.on_error)
        
        # Start worker
        self.worker.start()
        
    @Slot(np.ndarray, dict)
    def on_frame_ready(self, frame, metrics):
        """Display received frame"""
        # Convert numpy array to QPixmap
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        # Convert BGR to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Create QImage
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to pixmap and scale
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Display
        self.video_label.setPixmap(scaled)
        
    @Slot(int)
    def on_state_changed(self, state):
        """Update UI based on state"""
        state_name = StreamState(state).name
        self.status_label.setText(f"Status: {state_name}")
        
        # Update button states
        if state == StreamState.PAUSED:
            self.pause_button.setText("Resume")
        elif state == StreamState.RUNNING:
            self.pause_button.setText("Pause")
        elif state == StreamState.EDIT_ROI:
            self.roi_button.setText("Exit ROI Mode")
        
    @Slot(bool)
    def on_connection_changed(self, connected):
        """Update connection status"""
        if connected:
            self.video_label.setText("")
            self.status_label.setText("Status: Connected")
        else:
            self.video_label.setText("Disconnected")
            self.status_label.setText("Status: Disconnected")
            
    @Slot(str)
    def on_error(self, error_msg):
        """Display error message"""
        self.status_label.setText(f"Error: {error_msg}")
        
    def toggle_pause(self):
        """Toggle pause/resume"""
        if self.worker.state == StreamState.PAUSED:
            self.worker.resume()
        else:
            self.worker.pause()
            
    def toggle_roi_mode(self):
        """Toggle ROI edit mode"""
        if self.worker.state == StreamState.EDIT_ROI:
            self.worker.resume()
            self.roi_button.setText("Edit ROI Mode")
        else:
            self.worker.enter_roi_edit()
            self.roi_button.setText("Exit ROI Mode")
            
    def force_reconnect(self):
        """Force reconnection"""
        self.worker.set_state(StreamState.RECONNECTING)
        
    def closeEvent(self, event):
        """Clean shutdown"""
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    
    # Import cv2 for color conversion
    import cv2 as cv
    
    window = TestWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()