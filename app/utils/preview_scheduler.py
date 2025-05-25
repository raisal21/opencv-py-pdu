# preview_scheduler.py - Updated untuk thread safety
import logging
import cv2 as cv
import numpy as np
from PySide6.QtCore import QRunnable, Signal, QObject, QMetaObject, Qt, QMutex, QMutexLocker
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QSize
from models.camera import Camera

logger = logging.getLogger(__name__) 

class SnapshotSignal(QObject):
    """Pembungkus sinyal untuk dikirim balik ke GUI‑thread"""
    finished = Signal(int, np.ndarray)
    error = Signal(int, str)  # Tambahkan error signal

# Global mutex untuk mencegah multiple VideoCapture conflicts
_capture_mutex = QMutex()

class SnapshotWorker(QRunnable):
    """
    Thread-safe snapshot worker dengan proper error handling
    dan memory management
    """
    def __init__(self, cam_dict: dict[int, Camera], camera_id: int):
        super().__init__()
        self.camera_id = camera_id
        self.camera_dict = cam_dict
        self.signals = SnapshotSignal()

    def run(self):
        """Execute snapshot capture with comprehensive error handling"""
        cap = None
        try:
            # Dapatkan camera object dengan safe access
            if self.camera_id not in self.camera_dict:
                self.signals.error.emit(self.camera_id, "Camera not found in dictionary")
                return
                
            cam: Camera = self.camera_dict[self.camera_id]
            rtsp_url = cam.build_stream_url()
            
            # Validate URL sebelum membuat VideoCapture
            if not rtsp_url or not isinstance(rtsp_url, str):
                self.signals.error.emit(self.camera_id, "Invalid RTSP URL")
                return

            # Use mutex untuk mencegah multiple VideoCapture conflicts
            with QMutexLocker(_capture_mutex):
                # Create VideoCapture dengan timeout dan error handling
                cap = cv.VideoCapture(rtsp_url, cv.CAP_FFMPEG)

                if hasattr(cv, "CAP_PROP_THREAD_COUNT"):
                    cap.set(cv.CAP_PROP_THREAD_COUNT, 1)
                
                if not cap.isOpened():
                    self.signals.error.emit(self.camera_id, "Failed to open video capture")
                    return
                
                # Set properties dengan error checking
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 160)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 90)
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer untuk snapshot
                
                # Attempt to read frame dengan retry mechanism
                frame = None
                for attempt in range(3):  # Try up to 3 times
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                    elif attempt < 2:  # Don't sleep on last attempt
                        import time
                        time.sleep(0.1)  # Brief pause between attempts
                
                # Release VideoCapture immediately setelah capture
                cap.release()
                cap = None
                
                # Process frame jika berhasil
                if frame is not None and frame.size > 0:
                    self.signals.finished.emit(self.camera_id, frame.copy())
                else:
                    self.signals.error.emit(self.camera_id, "Failed to capture frame")
                    
        except Exception as e:
            error_msg = f"Snapshot error: {str(e)}"
            logger.error(f"Camera {self.camera_id}: {error_msg}")
            self.signals.error.emit(self.camera_id, error_msg)
        finally:
            # Ensure VideoCapture is always released
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass  # Ignore errors during cleanup

    def _safe_convert_to_pixmap(self, cv_frame: np.ndarray, target_size: QSize)
        """
        Thread-safe conversion dari OpenCV frame ke QPixmap
        dengan comprehensive error handling
        """
        try:
            # Validate input frame
            if cv_frame is None or cv_frame.size == 0:
                return QPixmap()
            
            # Ensure frame is in correct format (BGR)
            if len(cv_frame.shape) != 3 or cv_frame.shape[2] != 3:
                return QPixmap()
            
            # Convert BGR to RGB
            rgb_frame = cv.cvtColor(cv_frame, cv.COLOR_BGR2RGB)
            
            # Get frame dimensions
            height, width, channels = rgb_frame.shape
            bytes_per_line = channels * width
            
            # Create QImage dengan proper format
            qt_image = QImage(
                rgb_frame.data.tobytes(),  # Explicit conversion to bytes
                width, 
                height, 
                bytes_per_line, 
                QImage.Format_RGB888
            )
            
            # Validate QImage creation
            if qt_image.isNull():
                return QPixmap()
            
            # Convert to QPixmap
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to target size jika diperlukan
            if target_size.isValid() and not pixmap.isNull():
                pixmap = pixmap.scaled(
                    target_size, 
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            
            return pixmap
            
        except Exception as e:
            logger.error(f"Pixmap conversion error: {e}")
            return QPixmap()