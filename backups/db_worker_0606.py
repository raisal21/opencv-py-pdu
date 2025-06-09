from PySide6.QtCore import QRunnable, QObject, Signal, QMutex, QMutexLocker
from models.database import DatabaseManager
import threading

# Thread-local storage for database connections
_thread_local = threading.local()

# Mutex for connection creation
_db_mutex = QMutex()

def get_thread_db():
    """Get or create database connection for current thread"""
    if not hasattr(_thread_local, 'db'):
        with QMutexLocker(_db_mutex):
            _thread_local.db = DatabaseManager()
    return _thread_local.db

class DBSignals(QObject):
    finished = Signal(object)
    error = Signal(str)

class DBWorker(QRunnable):
    def __init__(self, method_name: str, *args, **kwargs):
        super().__init__()
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self.signals = DBSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            # Use thread-local connection
            db = get_thread_db()
            func = getattr(db, self.method_name)
            result = func(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))