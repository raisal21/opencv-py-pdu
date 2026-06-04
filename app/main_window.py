import sys
import logging
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QScrollArea, QSizePolicy, QMessageBox, QDialog
from shiboken6 import isValid
from PySide6.QtCore import Qt, QSize, QTimer, Signal, QThreadPool, Slot
from PySide6.QtGui import QIcon, QPixmap

from .resources import resource_path
from .models.camera import Camera, convert_cv_to_pixmap
from .views.add_camera import AddCameraDialog
from .views.camera_detail import CameraDetailUI
from .utils.log import setup as setup_log
from .utils.db_worker import DBWorker, DBSignals
from .utils.ping_scheduler import PingWorker
from .utils.preview_scheduler import PreviewScheduler

logger = logging.getLogger(__name__)

setup_log("--debug" in sys.argv)


class DeleteCameraDialog(QDialog):
    """Confirmation dialog for deleting a camera."""
    def __init__(self, camera_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Delete Camera")
        self.setFixedSize(400, 200)
        self.setStyleSheet("""
            QDialog {
                background-color: #09090B;
                border: 1px solid #27272A;
            }
            QLabel {
                color: #E4E4E7;
            }
            QPushButton {
                height: 35px;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 14px;
            }
        """)

        layout = QVBoxLayout(self)

        icon_label = QLabel()
        icon_label.setPixmap(QIcon(resource_path("assets/icons/warning.png")).pixmap(QSize(48, 48)))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        message = QLabel(f"Are you sure you want to delete this camera\n'{camera_name}'?\n\nThis action cannot be undone.")
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("font-size: 14px; margin: 5px;")
        layout.addWidget(message)

        button_layout = QHBoxLayout()

        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #3F3F46;
                color: #E4E4E7;
            }
            QPushButton:hover {
                background-color: #09090B;
                border: 1px solid #3F3F46;
            }
        """)
        cancel_button.clicked.connect(self.reject)

        delete_button = QPushButton("Delete")
        delete_button.setStyleSheet("""
            QPushButton {
                background-color: #DC2626;
                color: white;
            }
            QPushButton:hover {
                background-color: #991B1B;
            }
        """)
        delete_button.clicked.connect(self.accept)

        button_layout.addWidget(cancel_button)
        button_layout.addWidget(delete_button)

        layout.addLayout(button_layout)

class CameraItem(QFrame):
    """Camera row widget used in the camera list."""
    camera_clicked = Signal(int)
    edit_clicked = Signal(int)
    delete_clicked = Signal(int)

    def __init__(self, camera_id, camera_name, ip_address, port, protocol='RTSP',
                 username='', password='', stream_path='', url='',
                 is_online=False, preview_image=None, parent=None):
        super().__init__(parent)

        self.camera_id = camera_id
        self.camera_name = camera_name
        self.ip_address = ip_address
        self.port = port
        self.protocol = protocol
        self.username = username
        self.password = password
        self.stream_path = stream_path
        self.url = url
        self.is_online = is_online

        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QFrame {
                background-color: #09090B;
                border: 1px solid #27272A;
                border-radius: 4px;
                margin: 4px;
            }
            QFrame:hover {
                border: 1px solid #EA580C;
            }
        """)

        self.setCursor(Qt.PointingHandCursor)
        self.mousePressEvent = self.on_click

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.preview_widget = QLabel()
        self.preview_widget.setFixedSize(160, 90)
        self.preview_widget.setAlignment(Qt.AlignCenter)
        self.preview_widget.setStyleSheet("background-color: #1C1C1F; border-radius: 4px;")
        self.preview_widget.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        if preview_image:
            self.set_preview(preview_image)
        else:
            preview_text = "Offline" if not is_online else "Live"
            self.preview_widget.setText(preview_text)
            self.preview_widget.setStyleSheet(f"""
                background-color: #1C1C1F;
                color: {'#7f8c8d' if not is_online else '#4CAF50'};
                font-size: 14px;
                border-radius: 4px;
            """)

        layout.addWidget(self.preview_widget)
        layout.addSpacing(10)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)

        name_label = QLabel(camera_name)
        name_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #E4E4E7; border: 0px;")
        name_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        url_display = self.url if self.url else "[No URL Set]"
        if len(url_display) > 40:
            url_display = url_display[:37] + "..."

        ip_label = QLabel(f"URL: {url_display}")
        ip_label.setStyleSheet("font-size: 14px; color: #E4E4E7; border: 0px;")
        ip_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        status_color = "#4CAF50" if is_online else "#9CA3AF"
        status_text = "Online" if is_online else "Offline"
        self.status_label = QLabel(f"Status: {status_text}")
        self.status_label.setStyleSheet(f"font-size: 14px; color: {status_color}; border: 0px;")
        self.status_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        info_layout.addWidget(name_label)
        info_layout.addWidget(ip_label)
        info_layout.addWidget(self.status_label)
        info_layout.addStretch()

        layout.addLayout(info_layout, stretch=1)

        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)
        buttons_layout.setAlignment(Qt.AlignVCenter)

        buttons_container = QWidget()
        buttons_container.setFixedHeight(100)
        buttons_container.setStyleSheet("border: 0px;")
        buttons_inner_layout = QHBoxLayout(buttons_container)
        buttons_inner_layout.setContentsMargins(0, 0, 0, 0)
        buttons_inner_layout.setSpacing(16)

        edit_button = QPushButton()
        edit_button.setIcon(QIcon(resource_path("assets/icons/edit.png")))
        edit_button.setIconSize(QSize(20, 20))
        edit_button.setFixedSize(40, 40)
        edit_button.setStyleSheet("""
            QPushButton {
                background-color: #A1A1AA;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #EA580C;
            }
        """)
        edit_button.clicked.connect(self.on_edit_clicked)

        delete_button = QPushButton()
        delete_button.setIcon(QIcon(resource_path("assets/icons/trash.png")))
        delete_button.setIconSize(QSize(25, 25))
        delete_button.setFixedSize(40, 40)
        delete_button.setStyleSheet("""
            QPushButton {
                background-color: #A1A1AA;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #991B1B;
            }
        """)
        delete_button.clicked.connect(self.on_delete_clicked)

        buttons_inner_layout.addWidget(edit_button)
        buttons_inner_layout.addStretch()
        buttons_inner_layout.addWidget(delete_button)

        buttons_layout.addWidget(buttons_container)
        buttons_layout.addStretch()

        layout.addLayout(buttons_layout)


    def set_preview(self, image):
        """Set the camera preview image. Accepts a QPixmap or a BGR numpy array."""
        if isinstance(image, QPixmap):
            pixmap = image
        elif isinstance(image, np.ndarray):
            pixmap = convert_cv_to_pixmap(image, QSize(160, 90))
        else:
            return

        if not pixmap.isNull():
            self.preview_widget.setPixmap(pixmap)
            self.preview_widget.setStyleSheet("background-color: #1C1C1F; border-radius: 4px;")


    def update_status(self, is_online: bool):
        self.is_online = is_online
        status_color = "#4CAF50" if is_online else "#9CA3AF"
        status_text  = "Online"   if is_online else "Offline"
        self.status_label.setText(f"Status: {status_text}")
        self.status_label.setStyleSheet(f"font-size: 14px; color: {status_color}; border: 0px;")

        if not is_online:
            self.preview_widget.clear()
            self.preview_widget.setText("Offline")
            self.preview_widget.setStyleSheet(
                "background-color: #1C1C1F; color: #7f8c8d; font-size: 14px; border-radius: 4px;"
            )

    def on_click(self, event):
        """Open this camera item."""
        self.camera_clicked.emit(self.camera_id)

    def on_edit_clicked(self):
        """Emit the edit signal for this camera."""
        self.edit_clicked.emit(self.camera_id)

    def on_delete_clicked(self):
        """Emit the delete signal for this camera."""
        self.delete_clicked.emit(self.camera_id)

    def closeEvent(self, event):
        super().closeEvent(event)

class CameraList(QWidget):
    """Container widget for the camera list."""
    open_camera_detail = Signal(int)

    def __init__(self, error_handler, parent=None):
        super().__init__(parent)
        self.error_handler = error_handler
        self.active_cameras = {}
        self.snapshot_done = set()
        self.preview_cache = {}

        self.db_pool = QThreadPool.globalInstance()
        self.ping_pool   = QThreadPool.globalInstance()
        self.ping_pool.setMaxThreadCount(2)
        self.ping_timer  = QTimer(self)
        self.ping_timer.timeout.connect(self._refresh_statuses)
        self.ping_timer.start(30000)

        self.preview_pool = QThreadPool.globalInstance()
        self.preview_pool.setMaxThreadCount(2)
        self.preview_scheduler = PreviewScheduler(self.active_cameras)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(0)
        self.setStyleSheet("""
            QWidget {
                border: 1px solid #27272A;
            }
        """)

        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #27272A;
            }
        """)
        header_frame.setFixedHeight(30)

        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 0, 10, 0)

        header_label = QLabel("Camera List")
        header_label.setStyleSheet("font-size: 16px; color: #E4E4E7; font-weight: bold; border: 0px;")

        header_layout.addWidget(header_label)

        self.main_layout.addWidget(header_frame)

        self.cameras_container = QWidget()
        self.cameras_layout = QVBoxLayout(self.cameras_container)
        self.cameras_layout.setContentsMargins(0, 0, 0, 0)
        self.cameras_layout.setAlignment(Qt.AlignTop)
        self.cameras_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.cameras_container)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #27272A;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #27272A;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #27272A;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.main_layout.addWidget(scroll_area)

        self.empty_label = QLabel("No cameras found. Click 'Add Camera' to get started.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.cameras_layout.addWidget(self.empty_label)
        self.empty_label.setVisible(True)

        self.load_cameras()

    def load_cameras(self):
        """Load cameras from the database asynchronously."""
        self.ping_pool.clear()

        while self.cameras_layout.count():
            item = self.cameras_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.empty_label = QLabel(
            "No cameras found. Click 'Add Camera' to get started.",
            self.cameras_container
        )
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.cameras_layout.addWidget(self.empty_label)

        self.empty_label.setVisible(True)

        signals = DBSignals()
        signals.finished.connect(self._populate_camera_list)
        signals.error.connect(self.error_handler)

        worker = DBWorker(signals, "get_all_cameras")
        QThreadPool.globalInstance().start(worker)

    def _populate_camera_list(self, cameras):
        """Populate the camera list and request initial snapshots."""
        if not cameras:
            self._show_empty_state()
            return

        self._hide_empty_label()

        for camera_data in cameras:
            preview = self.preview_cache.get(camera_data['id'], None)
            camera_item = CameraItem(
                camera_id=camera_data['id'],
                camera_name=camera_data['name'],
                ip_address=camera_data['ip_address'],
                port=camera_data['port'],
                protocol=camera_data['protocol'],
                username=camera_data['username'],
                password=camera_data['password'],
                stream_path=camera_data['stream_path'],
                url=camera_data['url'],
                is_online=False,
                preview_image=preview if preview is not None else None
            )
            camera_item.camera_clicked.connect(self.open_camera_detail)
            camera_item.edit_clicked.connect(self.edit_camera)
            camera_item.delete_clicked.connect(self.delete_camera)
            self.cameras_layout.addWidget(camera_item)

            self.active_cameras[camera_data['id']] = Camera.from_dict(camera_data)

            self._request_initial_snapshot(camera_data['id'])

    def _request_initial_snapshot(self, camera_id: int):
        """Request the first snapshot for a camera once."""
        if camera_id in self.snapshot_done:
            return

        self.snapshot_done.add(camera_id)

        self.preview_scheduler.request_snapshot(
            camera_id,
            callback=self._on_snapshot_received,
        )

    def _hide_empty_label(self):
        """Hide the empty-state label when it still exists."""
        try:
            if (hasattr(self, "empty_label") and
                self.empty_label is not None and
                isValid(self.empty_label)):
                self.empty_label.setVisible(False)
        except RuntimeError:
            pass

    def _show_empty_state(self):
        """Show the empty-state label, recreating it if Qt already deleted it."""
        if not getattr(self, "empty_label", None) or not isValid(self.empty_label):
            self.empty_label = QLabel(
                "No cameras found. Click 'Add Camera' to get started.",
                self.cameras_container
            )
            self.empty_label.setAlignment(Qt.AlignCenter)
            self.cameras_layout.addWidget(self.empty_label)
        self.empty_label.setVisible(True)

    def add_camera(self, name, ip_address, port, protocol="RTSP", username="",
                password="", stream_path="", url="", roi_points=None):
        """Add a new camera to the database asynchronously."""
        signals = DBSignals()
        signals.finished.connect(self._get_added_camera_data)
        signals.error.connect(self.window()._db_error_msg)

        worker = DBWorker(
            signals,
            "add_camera",
            name, ip_address, port, protocol, username, password,
            stream_path, url, roi_points=roi_points
        )

        QThreadPool.globalInstance().start(worker)
        return True

    @Slot(int)
    def _get_added_camera_data(self, camera_id: int):
        """Load the full database row for a newly added camera."""
        if not camera_id:
            QMessageBox.warning(self, "Database Error", "Failed to get the new camera ID.")
            return

        signals = DBSignals()

        signals.finished.connect(self._on_camera_added)
        signals.error.connect(self.window()._db_error_msg)

        worker = DBWorker(signals, "get_camera", camera_id)
        QThreadPool.globalInstance().start(worker)

    @Slot(dict)
    def _on_camera_added(self, camera_data: dict):
        """Add the newly created camera to the UI."""
        if not camera_data:
            QMessageBox.warning(self, "Database Error", "Failed to load the newly added camera.")
            return
        if self.empty_label and self.empty_label.isVisible():
            self.empty_label.setVisible(False)

        camera_id = camera_data.get("id")
        if not camera_id:
            QMessageBox.warning(self, "DB Error", "Failed to add camera.")
            return

        if self.empty_label and self.empty_label.isVisible():
            self.empty_label.setVisible(False)

        self.active_cameras[camera_id] = Camera.from_dict(camera_data)

        camera_item = CameraItem(
            camera_id=camera_id,
            camera_name=camera_data['name'],
            ip_address=camera_data['ip_address'],
            port=camera_data['port'],
            protocol=camera_data['protocol'],
            username=camera_data['username'],
            password=camera_data['password'],
            stream_path=camera_data['stream_path'],
            url=camera_data['url'],
        )

        camera_item.camera_clicked.connect(self.open_camera_detail)
        camera_item.edit_clicked.connect(self.edit_camera)
        camera_item.delete_clicked.connect(self.delete_camera)
        self.cameras_layout.addWidget(camera_item)

        self._request_initial_snapshot(camera_id)

    def edit_camera(self, camera_id):
        """Load a camera before opening the edit dialog."""
        signals = DBSignals()
        signals.finished.connect(self._on_edit_camera_data_loaded)
        signals.error.connect(self.window()._db_error_msg)

        worker = DBWorker(signals, "get_camera", camera_id)
        QThreadPool.globalInstance().start(worker)

    @Slot(dict)
    def _on_edit_camera_data_loaded(self, camera_data):
        """Open the edit dialog after camera data has loaded."""
        if not camera_data:
            QMessageBox.warning(self, "Error", "Camera data not found.")
            return

        dialog = AddCameraDialog(self.window(), camera_data)
        if not dialog.exec():
            return

        updated = dialog.get_camera_data()

        update_signals = DBSignals()
        update_signals.finished.connect(lambda ok: self._on_camera_updated(ok, updated))
        update_signals.error.connect(self.window()._db_error_msg)

        worker = DBWorker(
            update_signals, "update_camera",
            camera_data['id'], updated['name'], updated['ip_address'], updated['port'],
            updated['protocol'], updated['username'], updated['password'],
            updated['stream_path'], updated['url']
        )
        QThreadPool.globalInstance().start(worker)

    def _on_camera_updated(self, success, updated):
        if success:
            self.load_cameras()
            QMessageBox.information(
                self.window(), "Camera Updated",
                f"Camera '{updated['name']}' has been updated successfully."
            )
        else:
            QMessageBox.warning(
                self.window(), "Update Failed",
                "Failed to update camera. Please try again."
            )

    def delete_camera(self, camera_id):
        """Load a camera before showing the delete confirmation."""
        signals = DBSignals()
        signals.finished.connect(self._on_delete_camera_data_loaded)
        signals.error.connect(self.window()._db_error_msg)

        worker = DBWorker(signals, "get_camera", camera_id)
        QThreadPool.globalInstance().start(worker)

    @Slot(dict)
    def _on_delete_camera_data_loaded(self, camera_data):
        """Show the delete confirmation after camera data has loaded."""
        if not camera_data: return

        dialog = DeleteCameraDialog(camera_data['name'], self.window())
        if not dialog.exec(): return

        delete_signals = DBSignals()
        delete_signals.finished.connect(lambda ok: self._on_camera_deleted(ok, camera_data['name'], camera_data['id']))
        delete_signals.error.connect(self.error_handler)

        worker = DBWorker(delete_signals, "delete_camera", camera_data['id'])
        QThreadPool.globalInstance().start(worker)

    def _on_camera_deleted(self, success, camera_name, camera_id):
        """Update local state after a camera has been deleted."""
        if success:
            self.active_cameras.pop(camera_id, None)
            self.preview_cache.pop(camera_id, None)
            self.load_cameras()
            QMessageBox.information(
                self.window(), "Camera Deleted",
                f"Camera '{camera_name}' has been deleted successfully."
            )
        else:
            QMessageBox.warning(
                self.window(), "Delete Failed",
                "Failed to delete camera. Please try again."
            )


    def _on_snapshot_received(self, camera_id: int, frame: np.ndarray):
        """Apply a received snapshot on the GUI thread."""
        found = False
        for i in range(self.cameras_layout.count()):
            widget = self.cameras_layout.itemAt(i).widget()
            if isinstance(widget, CameraItem) and widget.camera_id == camera_id:
                pixmap = convert_cv_to_pixmap(frame, QSize(160, 90))
                if not pixmap.isNull():
                    widget.set_preview(pixmap)
                    self.preview_cache[camera_id] = pixmap
                    widget.update_status(True)
                found = True
                break

        if not found:
            pixmap = convert_cv_to_pixmap(frame, QSize(160, 90))
            if not pixmap.isNull():
                self.preview_cache[camera_id] = pixmap

    def _update_item_status(self, camera_id: int, is_online: bool):
        """Update one CameraItem if the widget is still valid."""
        for i in range(self.cameras_layout.count()):
            widget = self.cameras_layout.itemAt(i).widget()
            if isinstance(widget, CameraItem) and widget.camera_id == camera_id:
                if isValid(widget):
                    widget.update_status(is_online)
                break

    def _refresh_statuses(self):
        """Ping cameras periodically and update their online status."""
        for i in range(self.cameras_layout.count()):
            widget = self.cameras_layout.itemAt(i).widget()
            if not isinstance(widget, CameraItem):
                continue


            worker = PingWorker(widget.camera_id, widget.ip_address, widget.port)
            worker.signals.finished.connect(self._update_item_status)
            self.ping_pool.start(worker)

    def showEvent(self, e):
        """Start background list updates when the widget is shown."""
        super().showEvent(e)

        if not self.ping_timer.isActive():
            self.ping_timer.start()

    def hideEvent(self, e):
        """Stop background list updates when the widget is hidden."""
        super().hideEvent(e)
        self.ping_timer.stop()

    def closeEvent(self, event):
        """Stop workers and timers before closing the list."""
        self.ping_timer.stop()

        if hasattr(self, 'preview_scheduler'):
            self.preview_scheduler.cancel_all()

        self.db_pool.waitForDone(2000)
        self.ping_pool.waitForDone(2000)
        self.preview_pool.waitForDone(2000)
        QThreadPool.globalInstance().waitForDone(2000)

        from .utils.preview_scheduler import _capture_pool
        _capture_pool.release_all()

        super().closeEvent(event)

class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()

        self.setup_ui()

        self.update_clock()
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        self.update_status_bar()

    def setup_ui(self):
        """Build the main window layout."""
        self.setWindowTitle("EyeLog - Camera Monitoring")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #09090B;")
        self.setWindowIcon(QIcon(resource_path("assets/icons/pdu.png")))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.Panel)
        header_frame.setFixedHeight(57)
        header_frame.setStyleSheet("background-color: #EA580C; border: 0px;")

        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 0, 20, 0)

        header_icon_label = QLabel()
        icon_pixmap = QIcon(resource_path("assets/icons/pdu.png")).pixmap(QSize(30, 30))
        if not icon_pixmap.isNull():
            header_icon_label.setPixmap(icon_pixmap)
        else:
            header_icon_label.setText("PDU")
            header_icon_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")

        header_title_label = QLabel("PDU | Parama Data Unit")
        header_title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        header_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        clock_icon_label = QLabel()
        clock_icon_pixmap = QIcon(resource_path("assets/icons/clock.png")).pixmap(QSize(18, 18))
        if not clock_icon_pixmap.isNull():
            clock_icon_label.setPixmap(clock_icon_pixmap)

        self.clock_text_label = QLabel()
        self.clock_text_label.setStyleSheet("font-size: 14px; color: #FFFFFF;")

        notification_button = QPushButton()
        notification_button.setFixedSize(35, 35)
        notification_button.setIcon(QIcon(resource_path("assets/icons/bell.png")))
        notification_button.setIconSize(QSize(20, 20))
        notification_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                padding: 6px 12px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #09090B;
            }
        """)
        notification_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        notification_button.clicked.connect(self.show_notification_dialog)

        header_layout.addWidget(header_icon_label)
        header_layout.addSpacing(4)
        header_layout.addWidget(header_title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(clock_icon_label)
        header_layout.addSpacing(3)
        header_layout.addWidget(self.clock_text_label)
        header_layout.addStretch(1)
        header_layout.addWidget(notification_button)

        navbar_frame = QFrame()
        navbar_frame.setFrameShape(QFrame.Panel)
        navbar_frame.setFixedHeight(55)
        navbar_frame.setStyleSheet("""
            QFrame {background-color: #09090B; border: 0px solid transparent; border-bottom: 4px solid #EA580C;}
        """)

        navbar_layout = QHBoxLayout(navbar_frame)
        navbar_layout.setContentsMargins(30, 0, 30, 0)

        navbar_title_label = QLabel("Video Realtime Camera Monitoring")
        navbar_title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: white; border: 0px;")

        self.add_camera_button = QPushButton("Add Camera")
        self.add_camera_button.setFixedHeight(35)
        self.add_camera_button.setStyleSheet("""
            QPushButton {
                background-color: #EA580C;
                color: #E4E4E7;
                font-size: 14px;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #09090B;
                border: 1px solid #EA580C;
            }
        """)
        self.add_camera_button.setIcon(QIcon(resource_path("assets/icons/plus.png")))
        self.add_camera_button.setIconSize(QSize(18, 18))
        self.add_camera_button.clicked.connect(self.show_add_camera_dialog)

        navbar_layout.addWidget(navbar_title_label)
        navbar_layout.addStretch(1)
        navbar_layout.addWidget(self.add_camera_button)

        self.camera_list = CameraList(error_handler=self._db_error_msg, parent=self)
        self.camera_list.open_camera_detail.connect(self.open_camera_detail)

        main_layout.addWidget(header_frame)
        main_layout.addWidget(navbar_frame)
        main_layout.addWidget(self.camera_list)

        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #27272A;
                color: #E4E4E7;
            }
        """)
        self.showMaximized()

    def update_clock(self):
        """Update the clock shown in the header."""
        current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        self.clock_text_label.setText(current_time)

    def show_notification_dialog(self):
        """Show the modal placeholder dialog for unavailable features."""
        QMessageBox.information(
            self,
            "Coming Soon",
            "This feature is under construction / not implemented yet.",
            QMessageBox.Ok
        )

    def update_status_bar(self):
        """Refresh the status bar with the current camera count."""
        signals = DBSignals()
        signals.finished.connect(lambda cameras: self.statusBar().showMessage(f"{len(cameras)} Cameras Connected | Database: Connected"))
        signals.error.connect(self._db_error_msg)

        worker = DBWorker(signals, "get_all_cameras")
        QThreadPool.globalInstance().start(worker)

    def pause_background_tasks(self):
        """Pause list background tasks while camera detail is open."""
        if hasattr(self, 'clock_timer') and self.clock_timer.isActive():
            self.clock_timer.stop()

        if hasattr(self, 'camera_list') and hasattr(self.camera_list, 'ping_timer'):
            if self.camera_list.ping_timer.isActive():
                self.camera_list.ping_timer.stop()

        if hasattr(self.camera_list, 'ping_pool'):
            self.camera_list.ping_pool.clear()

    def resume_background_tasks(self):
        """Resume list background tasks after returning from camera detail."""
        if hasattr(self, 'clock_timer'):
            self.clock_timer.start(1000)
            self.update_clock()

        if hasattr(self, 'camera_list') and hasattr(self.camera_list, 'ping_timer'):
            self.camera_list.ping_timer.start(30000)

    def show_add_camera_dialog(self):
        """Open the dialog for adding a camera."""
        dialog = AddCameraDialog(self)

        if dialog.exec():
            camera_data = dialog.get_camera_data()

            self.camera_list.add_camera(
                camera_data['name'], camera_data['ip_address'], camera_data['port'],
                camera_data['protocol'], camera_data['username'], camera_data['password'],
                camera_data['stream_path'], camera_data['url'],
                roi_points=camera_data.get('roi_points')
            )

    def open_camera_detail(self, camera_id):
        """Load a camera asynchronously and open its detail page."""
        signals = DBSignals()

        @Slot(dict)
        def _on_data_received(camera_data):
            if camera_data:
                self.pause_background_tasks()

                detail_window = CameraDetailUI(camera_data, parent=self)
                detail_window.destroyed.connect(self.resume_background_tasks)

                self.hide()
                detail_window.show()
            else:
                QMessageBox.warning(self, "Camera Not Found", f"Camera with ID {camera_id} not found.")

        signals.finished.connect(_on_data_received)
        signals.error.connect(self._db_error_msg)

        worker = DBWorker(signals, "get_camera", camera_id)
        QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _db_error_msg(self, error_message: str):
        """Show a database error through a single handler."""
        logger.error(f"Database operation failed: {error_message}")
        QMessageBox.critical(self, "Database Error", error_message)

    def closeEvent(self, event):
        """Clean up background work before the application closes."""
        if hasattr(self, 'clock_timer'):
            self.clock_timer.stop()

        if hasattr(self, 'camera_list'):
            self.camera_list.close()

        super().closeEvent(event)

