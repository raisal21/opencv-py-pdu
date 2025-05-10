# app/utils/frame_processor.py
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
import numpy as np

from utils.material_detector import ForegroundExtraction, ContourProcessor
from models.camera import Camera   # hanya tipe; tidak mem‑pull Qt apa pun


class FrameProcessor(QObject):
    """
    Worker yang menjalankan semua CPU‑bound OpenCV di thread terpisah.
    """
    processed = Signal(np.ndarray, dict)          # (frame_tampil, metrics)

    def __init__(self, camera: Camera,
                 bg_params: dict,
                 contour_params: dict,
                 parent=None):
        super().__init__(parent)
        self.camera = camera
        self.bg_subtractor = ForegroundExtraction(**bg_params)
        self.contour_proc  = ContourProcessor(**contour_params)

    # ----------------------- PUBLIC SLOT -----------------------

    @Slot(np.ndarray)
    def process(self, frame: np.ndarray):
        """Dipanggil dari sinyal CameraThread.frame_received (queued)."""
        # 1) Terapkan ROI ringan dari objek Camera
        roi_frame = self.camera.process_frame_with_roi(frame)
        if roi_frame is None:
            return

        try:
            # 2) Background‑subtraction & contour
            fg = self.bg_subtractor.process_frame(roi_frame)
            cnt = self.contour_proc.process_mask(fg.binary)

            display = self.contour_proc.visualize(
                roi_frame, cnt.contours, cnt.metrics
            )
            # 3) Kirim balik ke UI
            self.processed.emit(display, cnt.metrics)

        except Exception:         # log seperlunya; jangan crash worker
            self.processed.emit(roi_frame, {})
