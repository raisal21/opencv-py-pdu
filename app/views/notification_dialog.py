from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QLabel
from PySide6.QtCore    import Qt
from resources import resource_path

class NotificationDialog(QDialog):
    """
    Dialog sederhana yang menampilkan daftar notifikasi.  Item bersifat
    clickable; pada klik dialog ditutup & sinyal `notification_clicked`
    di-emit agar MainWindow membuka Camera Detail.
    """
    def __init__(self, notifications: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Notifications")
        self.setFixedSize(400, 500)
        self.setStyleSheet("""
            QDialog {background-color:#09090B; border:1px solid #27272A;}
            QListWidget {background-color:#1C1C1F; color:#E4E4E7; border:0px;}
            QListWidget::item:selected {background:#27272A;}
            QLabel {color:#E4E4E7;}
        """)

        layout = QVBoxLayout(self)
        if not notifications:
            layout.addWidget(QLabel("No notifications."))
            return

        self.list_widget = QListWidget()
        for n in notifications:
            text = f"[{n['time']}] {n['camera_name']}  |  {n['message']}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, n["camera_id"])
            self.list_widget.addItem(item)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    # ----------------------------------------------------------
    def _on_item_clicked(self, item):
        cam_id = item.data(Qt.UserRole)
        self.accept()
        # minta parent (MainWindow) buka detail
        if self.parent() and hasattr(self.parent(), "open_camera_detail"):
            self.parent().open_camera_detail(cam_id)
