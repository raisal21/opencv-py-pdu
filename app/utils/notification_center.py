from datetime import datetime
from PySide6.QtCore import QObject, Signal

class NotificationCenter(QObject):
    """
    Menyimpan notifikasi di memori (max-beberapa KB) dan mem-broadcast
    setiap penambahan notifikasi baru.
    """
    new_notification = Signal(dict)  # {'camera_id', 'camera_name', 'message', 'time'}

    def __init__(self, parent=None, max_items=20):
        super().__init__(parent)
        self._items      = []           # newest first
        self._max_items  = max_items

    # --------------------------------------------------------------
    def add(self, camera_id: int, message: str, camera_name: str):
        now = datetime.now().strftime("%H:%M:%S")
        data = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "message": message,
            "time": now,
        }
        # sisip di depan (terbaru)
        self._items.insert(0, data)
        # trim bila perlu
        if len(self._items) > self._max_items:
            self._items.pop()
        self.new_notification.emit(data)

    def all(self):
        """Kembalikan copy daftar notifikasi (paling baru di indeks 0)."""
        return list(self._items)
