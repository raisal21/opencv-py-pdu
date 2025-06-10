import traceback
import logging
import threading
from PySide6.QtCore import QRunnable, QObject, Signal, QMutex, QMutexLocker, Slot
from ..models.database import DatabaseManager, DB_PATH

logger = logging.getLogger(__name__)

# Thread-local storage for database connections
_thread_local = threading.local()

# Mutex for connection creation
_db_mutex = QMutex()

def get_thread_db():
    """Get or create a database connection for the current thread."""
    if not hasattr(_thread_local, 'db'):
        with QMutexLocker(_db_mutex):
            # Double-check inside the lock to prevent race conditions
            if not hasattr(_thread_local, 'db'):
                _thread_local.db = DatabaseManager()
    return _thread_local.db

class DBSignals(QObject):
    """
    Defines the signals available from a database worker.
    - finished: Emitted when the task is completed. The 'object' can be any result.
    - error: Emitted when an error occurs.
    """
    finished = Signal(object)
    error = Signal(str)

class DBWorker(QRunnable):
    """
    A generic QRunnable worker for executing database operations in a background thread.
    It takes a signals object to communicate back to the main thread.
    """

    _db_manager = DatabaseManager(db_path=DB_PATH)

    def __init__(self, signals: DBSignals, method_name: str, *args, **kwargs):
        """
        Initializes the worker.
        
        Args:
            signals (DBSignals): The signal object to use for communication.
            method_name (str): The name of the method to call on DatabaseManager.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.
        """
        super().__init__()
        self.signals = signals
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        """
        Menjalankan siklus lengkap: buka koneksi, jalankan operasi,
        commit jika perlu, dan tutup koneksi.
        """
        conn = None
        try:
            # 1. Buka koneksi baru khusus untuk thread ini
            conn = self._db_manager.get_connection()
            
            # (Opsional tapi direkomendasikan) Pastikan tabel ada
            self._db_manager.ensure_tables(conn)

            # 2. Dapatkan metode dari instance manager
            method_to_call = getattr(self._db_manager, self.method_name)
            
            # 3. Jalankan metode dengan koneksi sebagai argumen pertama
            result = method_to_call(conn, *self.args, **self.kwargs)

            conn.commit()
            logger.info(f"DB transaction committed for method: {self.method_name}")

            # 5. Kirim sinyal bahwa tugas selesai
            self.signals.finished.emit(result)
            
        except Exception as e:
            if conn:
                conn.rollback() # Batalkan perubahan jika terjadi error
            error_message = f"Database worker failed for method '{self.method_name}': {e}\n{traceback.format_exc()}"
            logger.error(error_message)
            self.signals.error.emit(error_message)
        finally:
            # 6. Pastikan koneksi selalu ditutup
            if conn:
                conn.close()
                logger.debug(f"DB connection closed for thread.")

