from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
import numpy as np
import time
from utils.material_detector import ForegroundExtraction, ContourProcessor
from models.camera import Camera

class FrameProcessor(QObject):
    processed = Signal(np.ndarray, dict)
    
    def __init__(self, camera: Camera, bg_params: dict, contour_params: dict, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.bg_subtractor = ForegroundExtraction(**bg_params)
        self.contour_proc = ContourProcessor(**contour_params)
        
        # Throttling
        self._last_process_time = 0
        self._min_process_interval = 0.1  # 100ms minimum between processing
        self._frame_skip_count = 0
        self._max_frame_skip = 2  # Process every 3rd frame maximum

    @Slot(np.ndarray)
    def process(self, frame: np.ndarray):
        # Throttle processing
        current_time = time.time()
        time_diff = current_time - self._last_process_time
        
        # Skip frame if too soon
        if time_diff < self._min_process_interval:
            self._frame_skip_count += 1
            if self._frame_skip_count < self._max_frame_skip:
                return
        
        self._frame_skip_count = 0
        self._last_process_time = current_time
        
        # Process frame
        roi_frame = self.camera.process_frame_with_roi(frame)
        if roi_frame is None:
            return

        try:
            fg = self.bg_subtractor.process_frame(roi_frame)
            cnt = self.contour_proc.process_mask(fg.binary)
            display = self.contour_proc.visualize(roi_frame, cnt.contours, cnt.metrics)
            self.processed.emit(display, cnt.metrics)
        except Exception as e:
            # Log but don't crash
            import logging
            logging.error(f"Frame processing error: {e}")
            self.processed.emit(roi_frame, {})