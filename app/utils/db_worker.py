# models/db_worker.py
from PySide6.QtCore import QRunnable, QObject, Signal
from models.database import DatabaseManager

class DBSignals(QObject):
    finished = Signal(object)     # payload hasil (bisa None)
    error    = Signal(str)

class DBWorker(QRunnable):
    """
    Jalankan satu method di DatabaseManager di thread‑pool
    supaya UI tidak nge‑lag saat operasi I/O.
    """
    def __init__(self, method_name: str, *args, **kwargs):
        super().__init__()
        self.method_name = method_name
        self.args        = args
        self.kwargs      = kwargs
        self.signals     = DBSignals()

    def run(self):
        try:
            db = DatabaseManager()          # koneksi BARU → thread‑safe
            func = getattr(db, self.method_name)
            result = func(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
