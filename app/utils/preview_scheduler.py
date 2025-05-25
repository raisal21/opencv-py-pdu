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
        """Execute snapshot capture - hanya numpy array, tanpa Qt GUI objects"""
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
                
                # Set properties untuk snapshot kecil
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)   # Lebih besar untuk kualitas lebih baik
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 180)  # Maintain 16:9 ratio
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                
                # Attempt to read frame dengan retry mechanism
                frame = None
                max_attempts = 3
                
                for attempt in range(max_attempts):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                    elif attempt < max_attempts - 1:
                        time.sleep(0.1)  # Brief pause between attempts
                
                # Release VideoCapture immediately
                cap.release()
                cap = None
                
                # PENTING: Kirim numpy array, BUKAN QPixmap!
                if frame is not None and frame.size > 0:
                    # Resize frame untuk preview jika terlalu besar
                    if frame.shape[1] > 160 or frame.shape[0] > 90:
                        frame = cv.resize(frame, (160, 90), interpolation=cv.INTER_AREA)
                    
                    # Emit numpy array (copy untuk thread safety)
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
                    pass
                

class PreviewScheduler(QObject):
    """
    Scheduler untuk mengatur preview updates dengan thread-safe approach.
    Konversi ke QPixmap dilakukan di GUI thread.
    """
    def __init__(self, camera_dict: dict[int, Camera], parent=None):
        super().__init__(parent)
        self.camera_dict = camera_dict
        self.pending_workers = set()  # Track active workers
        
    def request_snapshot(self, camera_id: int, callback=None, error_callback=None):
        """
        Request snapshot untuk camera tertentu.
        
        Args:
            camera_id: ID camera
            callback: Fungsi yang dipanggil dengan (camera_id, numpy_array)
            error_callback: Fungsi yang dipanggil dengan (camera_id, error_msg)
        """
        if camera_id in self.pending_workers:
            return  # Skip if already processing
            
        worker = SnapshotWorker(self.camera_dict, camera_id)
        
        # Connect callbacks
        if callback:
            worker.signals.finished.connect(
                lambda cid, frame: self._handle_snapshot(cid, frame, callback)
            )
        
        if error_callback:
            worker.signals.error.connect(error_callback)
            
        # Track worker
        self.pending_workers.add(camera_id)
        worker.signals.finished.connect(
            lambda cid, _: self.pending_workers.discard(cid)
        )
        worker.signals.error.connect(
            lambda cid, _: self.pending_workers.discard(cid)
        )
        
        # Start worker
        from PySide6.QtCore import QThreadPool
        QThreadPool.globalInstance().start(worker)
    
    def _handle_snapshot(self, camera_id: int, frame: np.ndarray, callback):
        """Handle snapshot result di GUI thread"""
        # Di sini kita di GUI thread, jadi aman untuk convert ke QPixmap
        # jika callback membutuhkannya
        callback(camera_id, frame)