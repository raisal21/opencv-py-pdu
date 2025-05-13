# preview_scheduler.py
import logging
import cv2 as cv

from PySide6.QtCore import QRunnable, Signal, QObject, QMetaObject, Qt
from PySide6.QtGui  import QImage, QPixmap
from PySide6.QtCore import QSize
from models.camera import Camera, convert_cv_to_pixmap

logger = logging.getLogger(__name__) 

class SnapshotSignal(QObject):
    """Pembungkus sinyal untuk dikirim balik ke GUI‑thread"""
    finished = Signal(int, QPixmap)

class SnapshotWorker(QRunnable):
    """
    Mengambil 1 frame 160×90 tanpa membuka stream permanen.
    Dipakai di QThreadPool → otomatis reuse thread.
    """
    def __init__(self, cam_dict: dict[int, Camera], camera_id: int):
        super().__init__()
        self.camera_id   = camera_id
        self.camera_dict = cam_dict           # akses ke data kamera (url dll.)
        self.signals     = SnapshotSignal()

    def run(self):
        cam: Camera = self.camera_dict[self.camera_id]
        rtsp_url = cam.build_stream_url()

        # OpenCV snapshot
        cap = cv.VideoCapture(rtsp_url, cv.CAP_FFMPEG)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 160)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 90)
        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            pixmap = convert_cv_to_pixmap(frame, QSize(160, 90))
            self.signals.finished.emit(self.camera_id, pixmap)
