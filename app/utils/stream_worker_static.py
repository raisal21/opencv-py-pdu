import time
import cv2 as cv
import numpy as np
import logging
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from ..resources import resource_path # Impor untuk mendapatkan path absolut
from ..utils.material_detector_static import ForegroundExtraction, ContourProcessor, BG_PRESETS, CONTOUR_PRESETS

logger = logging.getLogger(__name__)

class StreamWorkerStatic(QThread):
    """
    Worker yang disederhanakan untuk memproses video dari file lokal secara berulang.
    MODIFIED: Menambahkan kemampuan untuk memperbarui preset secara dinamis.
    """
    frame_ready = Signal(np.ndarray, dict)
    error_occurred = Signal(str)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._stop_flag = False
        self._mutex = QMutex()

        # Mutex untuk melindungi akses ke parameter preset
        self._params_mutex = QMutex()
        self._pending_bg_params = None
        self._pending_contour_params = None
        
        # Inisialisasi prosesor dengan preset default
        self._initialize_processors(BG_PRESETS["default"], CONTOUR_PRESETS["standard"])

    def _initialize_processors(self, bg_params, contour_params):
        """Inisialisasi atau re-inisialisasi prosesor deteksi."""
        self.bg_subtractor = ForegroundExtraction(**bg_params)
        self.contour_processor = ContourProcessor(**contour_params)
        logger.info("Detection processors initialized/updated.")

    def run(self):
        """Loop utama thread untuk membaca dan memproses video."""
        video_path_abs = resource_path(self.camera.video_path)
        cap = cv.VideoCapture(video_path_abs)

        if not cap.isOpened():
            self.error_occurred.emit(f"Gagal membuka file video: {video_path_abs}")
            return

        while not self._is_stopping():
            # Terapkan pembaruan preset sebelum memproses frame berikutnya
            self._apply_pending_preset_updates()

            ret, frame = cap.read()
            if not ret:
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue

            processed_frame, metrics = self._process_frame(frame)
            self.frame_ready.emit(processed_frame, metrics)
            
            time.sleep(1 / 15) # Target ~15 FPS

        cap.release()
        logger.info("Static StreamWorker stopped.")

    def _apply_pending_preset_updates(self):
        """Secara thread-safe menerapkan parameter preset yang baru."""
        bg_params_to_apply = None
        contour_params_to_apply = None

        with QMutexLocker(self._params_mutex):
            if self._pending_bg_params:
                bg_params_to_apply = self._pending_bg_params
                self._pending_bg_params = None
            
            if self._pending_contour_params:
                contour_params_to_apply = self._pending_contour_params
                self._pending_contour_params = None
        
        # Re-inisialisasi di luar lock untuk menghindari deadlock
        if bg_params_to_apply or contour_params_to_apply:
            # Gunakan parameter yang ada jika salah satunya tidak diupdate
            current_bg_params = bg_params_to_apply or self.bg_subtractor.get_params()
            current_contour_params = contour_params_to_apply or self.contour_processor.get_params()
            self._initialize_processors(current_bg_params, current_contour_params)

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

    def set_bg_params(self, params: dict):
        """Metode publik untuk UI thread mengatur parameter BG baru."""
        with QMutexLocker(self._params_mutex):
            self._pending_bg_params = params

    def set_contour_params(self, params: dict):
        """Metode publik untuk UI thread mengatur parameter Contour baru."""
        with QMutexLocker(self._params_mutex):
            self._pending_contour_params = params

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop_flag = True

    def _is_stopping(self):
        with QMutexLocker(self._mutex):
            return self._stop_flag

