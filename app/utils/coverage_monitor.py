# utils/coverage_monitor.py
import time, json
from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer
from utils.frame_processor import FrameProcessor            # sudah ada
from utils.coverage_logger import CoverageLogger            # sudah ada
from models.camera import Camera, FPS_IDLE

class CoverageMonitor(QObject):
    danger = Signal(int, float)          # camera_id, percent

    def __init__(self, camera: Camera,
                 thresh=85,                # % “bahaya”
                 notif_delay=300,          # 5 menit
                 log_interval=300,         # 5 menit
                 parent=None):
        super().__init__(parent)
        self.cam           = camera
        self.thresh        = thresh
        self.notif_delay   = notif_delay
        self.log_interval  = log_interval
        self._last_notif   = 0
        self._last_save    = 0
        self.logger        = CoverageLogger()

        # ----- siapkan thread analisis ringan -----
        self.cam.is_preview_mode = True
        self.cam.start_stream()                # FPS_IDLE otomatis
        self._worker_th = QThread(self)
        bg_params       = {"history": 200}     # preset sangat ringan
        contour_params  = {"min_contour_area": 150}
        self.proc       = FrameProcessor(self.cam, bg_params, contour_params)
        self.proc.moveToThread(self._worker_th)
        self.cam.thread.frame_received.connect(
            self.proc.process,        Qt.QueuedConnection)
        self.proc.processed.connect(self._on_metrics, Qt.QueuedConnection)
        self._worker_th.start()

    # --------------------------------------------
    def _on_metrics(self, _frame, metrics: dict):
        cov = metrics.get("processed_coverage_percent", 0.0)
        now = time.time()

        # --- notifikasi bahaya ---
        if cov >= self.thresh and (now - self._last_notif) >= self.notif_delay:
            self.danger.emit(self.cam.id, cov)
            self._last_notif = now

        # --- simpan CSV setiap 5′ saja ---
        if (now - self._last_save) >= self.log_interval:
            self.logger.log_measurement(self.cam.id, cov)
            self._last_save = now

    def stop(self):
        """Matikan thread & putus semua sinyal (aman dipanggil berulang)."""
        try:
            self.cam.thread.frame_received.disconnect(self.proc.process)
        except (RuntimeError, TypeError):
            pass

        try:
            self.proc.processed.disconnect(self._on_metrics)
        except (RuntimeError, TypeError):
            pass

        # hentikan stream kalau masih hidup
        if hasattr(self.cam, "stop_stream"):
            self.cam.stop_stream()

        self._worker_th.quit()
        self._worker_th.wait(1000)


