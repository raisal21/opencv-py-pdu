# app/utils/preview_scheduler.py
# COMPLETE FIXED VERSION

import logging
import cv2 as cv
import numpy as np
import time
import platform
from PySide6.QtCore import QRunnable, Signal, QObject, QMutex, QMutexLocker
from models.camera import Camera

logger = logging.getLogger(__name__)

# Global mutex untuk mencegah multiple VideoCapture conflicts
_capture_mutex = QMutex()

# Connection pool untuk reuse VideoCapture
class VideoCapturePool:
    def __init__(self, max_size=3):
        self.pool = {}
        self.max_size = max_size
        self.mutex = QMutex()
        self.last_used = {}
    
    def get_capture(self, url, *args, **kwargs):
        # Ignore extra arguments for backward compatibility
        with QMutexLocker(self.mutex):
            # Check if exists in pool
            if url in self.pool:
                self.last_used[url] = time.time()
                return self.pool[url]
            
            # Create new if pool not full
            if len(self.pool) < self.max_size:
                cap = self._create_capture(url)
                if cap and cap.isOpened():
                    self.pool[url] = cap
                    self.last_used[url] = time.time()
                    return cap
            
            # Release oldest and create new
            oldest_url = min(self.last_used, key=self.last_used.get)
            old_cap = self.pool.pop(oldest_url)
            try:
                old_cap.release()
            except:
                pass
            del self.last_used[oldest_url]
            
            # Create new
            cap = self._create_capture(url)
            if cap and cap.isOpened():
                self.pool[url] = cap
                self.last_used[url] = time.time()
                return cap
            
            return None
    
    def _create_capture(self, url):
        try:
            cap = cv.VideoCapture(url, cv.CAP_FFMPEG)
            
            # Platform specific settings
            if platform.system() == "Windows":
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 180)
            elif platform.system() == "Darwin":  # macOS
                cap.set(cv.CAP_PROP_HW_ACCELERATION, 0)
                
            return cap
        except Exception as e:
            logger.error(f"Failed to create capture: {e}")
            return None
    
    def release_all(self):
        with QMutexLocker(self.mutex):
            for cap in self.pool.values():
                try:
                    cap.release()
                except:
                    pass
            self.pool.clear()
            self.last_used.clear()

# Global pool instance
_capture_pool = VideoCapturePool()


class SnapshotSignal(QObject):
    finished = Signal(int, np.ndarray)
    error = Signal(int, str)


class SnapshotWorker(QRunnable):
    def __init__(self, cam_dict: dict[int, Camera], camera_id: int):
        super().__init__()
        self.camera_id = camera_id
        self.camera_dict = cam_dict
        self.signals = SnapshotSignal()
        self.setAutoDelete(True)  # Important for cleanup

    def run(self):
        cap = None
        try:
            if self.camera_id not in self.camera_dict:
                self.signals.error.emit(self.camera_id, "Camera not found")
                return
                
            cam = self.camera_dict[self.camera_id]
            rtsp_url = cam.build_stream_url()
            
            if not rtsp_url:
                self.signals.error.emit(self.camera_id, "Invalid URL")
                return

            # Use connection pool
            with QMutexLocker(_capture_mutex):
                cap = _capture_pool.get_capture(rtsp_url)
                
                if not cap:
                    self.signals.error.emit(self.camera_id, "Failed to open capture")
                    return
                
                # Read with timeout
                frame = None
                max_attempts = 3
                
                for attempt in range(max_attempts):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                    time.sleep(0.05)
                
                # Don't release - pool will manage it
                cap = None
                
                if frame is not None and frame.size > 0:
                    # Resize for preview
                    frame_resized = cv.resize(frame, (160, 90), interpolation=cv.INTER_AREA)
                    self.signals.finished.emit(self.camera_id, frame_resized.copy())
                else:
                    self.signals.error.emit(self.camera_id, "Failed to capture frame")
                    
        except Exception as e:
            logger.error(f"Snapshot error for camera {self.camera_id}: {str(e)}")
            self.signals.error.emit(self.camera_id, str(e))


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