from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QFrame, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
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
            QDialog {
                background-color: #09090B; 
                border: 1px solid #27272A;
            }
            QListWidget {
                background-color: #1C1C1F; 
                color: #E4E4E7; 
                border: 0px;
            }
            QListWidget::item:selected {
                background: #27272A;
            }
            QListWidget::item {
                padding: 8px 4px;
            }
            QLabel {
                color: #E4E4E7;
            }
        """)

        # Create main layout without margins
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create orange header frame
        header_frame = QFrame()
        header_frame.setFixedHeight(40)
        header_frame.setStyleSheet("background-color: #EA580C; border: 0px;")
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 0, 15, 0)
        
        # Header title
        header_title = QLabel("Notifications")
        header_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        
        header_layout.addWidget(header_title)
        header_layout.addStretch()
        
        # Add header to main layout
        main_layout.addWidget(header_frame)
        
        # Content area
        content_frame = QFrame()
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(10, 10, 10, 10)
        
        if not notifications:
            no_notifications_label = QLabel("No notifications.")
            no_notifications_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            no_notifications_label.setContentsMargins(8, 8, 8, 8)
            content_layout.addWidget(no_notifications_label, 0, Qt.AlignTop)
        else:
            self.list_widget = QListWidget()
            self.list_widget.setAlternatingRowColors(True)
            self.list_widget.setStyleSheet("""
                QListWidget {
                    alternating-background-color: #242427;
                }
            """)
            
            for n in notifications:
                text = f"[{n['time']}] {n['camera_name']}  |  {n['message']}"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, n["camera_id"])
                self.list_widget.addItem(item)
            
            self.list_widget.itemClicked.connect(self._on_item_clicked)
            content_layout.addWidget(self.list_widget)
        
        main_layout.addWidget(content_frame, 1)

    def _on_item_clicked(self, item):
        cam_id = item.data(Qt.UserRole)
        self.accept()
        # minta parent (MainWindow) buka detail
        if self.parent() and hasattr(self.parent(), "open_camera_detail"):
            self.parent().open_camera_detail(cam_id)