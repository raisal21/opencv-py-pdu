import time
import cv2 as cv
import numpy as np
import logging
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from ..utils.material_detector import ForegroundExtraction, ContourProcessor, BG_PRESETS, CONTOUR_PRESETS

logger = logging.getLogger(__name__)

class StreamWorkerStatic(QThread):
    """
    Worker yang disederhanakan untuk memproses video dari file lokal secara berulang (loop).
    """
    frame_ready = Signal(np.ndarray, dict)  # (processed_frame, metrics)
    error_occurred = Signal(str)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._stop_flag = False
        self._mutex = QMutex()
        
        # Gunakan preset default untuk demo
        self.bg_subtractor = ForegroundExtraction(**BG_PRESETS["default"])
        self.contour_processor = ContourProcessor(**CONTOUR_PRESETS["standard"])

    def run(self):
        video_path = self.camera.video_path
        cap = cv.VideoCapture(video_path)

        if not cap.isOpened():
            self.error_occurred.emit(f"Gagal membuka file video: {video_path}")
            return

        while not self._is_stopping():
            ret, frame = cap.read()
            if not ret:
                # Jika video selesai, putar ulang dari awal
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue

            # Proses frame
            processed_frame, metrics = self._process_frame(frame)
            self.frame_ready.emit(processed_frame, metrics)
            
            # Kontrol FPS sederhana
            time.sleep(1 / 15) # Target ~15 FPS

        cap.release()
        logger.info("Static StreamWorker stopped.")

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Memproses satu frame video."""
        roi_frame = self.camera.process_frame_with_roi(frame)
        if roi_frame is None:
            return frame, {}
            
        fg_result = self.bg_subtractor.process_frame(roi_frame)
        contour_result = self.contour_processor.process_mask(fg_result.binary)
        
        display_frame = self.contour_processor.visualize(
            roi_frame, contour_result.contours, contour_result.metrics, show_metrics=False
        )
        return display_frame, contour_result.metrics

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop_flag = True

    def _is_stopping(self):
        with QMutexLocker(self._mutex):
            return self._stop_flag
