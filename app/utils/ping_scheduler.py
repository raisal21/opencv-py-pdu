# ping_scheduler.py
import logging
from PySide6.QtCore import QRunnable, Signal, QObject
from views.add_camera import validate_ip_address

logger = logging.getLogger(__name__) 


class PingSignal(QObject):
    finished = Signal(int, bool)        # (camera_id, is_online)

class PingWorker(QRunnable):
    """TCP‑ping ringan (≤ 500 ms timeout) memakai thread‑pool global."""
    def __init__(self, camera_id, ip, port, timeout=0.5):
        super().__init__()
        self.camera_id = camera_id
        self.ip        = ip
        self.port      = port
        self.timeout   = timeout
        self.signals   = PingSignal()

    def run(self):
        ok = validate_ip_address(self.ip, self.port, self.timeout)
        self.signals.finished.emit(self.camera_id, ok)
