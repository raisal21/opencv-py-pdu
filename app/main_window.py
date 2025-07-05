import sys
import os
import json
import time
import logging
from typing import Any
import numpy as np
import cv2 as cv
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QFrame, QScrollArea, QSizePolicy, 
                              QMessageBox, QDialog)
from shiboken6 import isValid
from PySide6.QtCore import Qt, QSize, QTimer, Signal, QObject, QThread, QThreadPool, Slot
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap

from .resources import resource_path
from .models.camera import Camera, convert_cv_to_pixmap, CameraThread
from .views.add_camera import AddCameraDialog
from .views.camera_detail import CameraDetailUI
from .views.camera_detail_static import CameraDetailUI_Static
from .utils.log import setup as setup_log
from .utils.db_worker import DBWorker, DBSignals
from .utils.ping_scheduler import PingWorker
from .utils.preview_scheduler import SnapshotWorker, PreviewScheduler

logger = logging.getLogger(__name__)

setup_log("--debug" in sys.argv)


class DeleteCameraDialog(QDialog):
    """Dialog konfirmasi penghapusan kamera"""
    def __init__(self, camera_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Konfirmasi Hapus")
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
        
        # Ikon peringatan   
        icon_label = QLabel()
        icon_label.setPixmap(QIcon(resource_path("assets/icons/warning.png")).pixmap(QSize(48, 48)))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        # Pesan konfirmasi
        message = QLabel(f"Apakah Anda yakin ingin menghapus kamera\n'{camera_name}'?\n\nTindakan ini tidak dapat dibatalkan.")
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("font-size: 14px; margin: 5px;")
        layout.addWidget(message)
        
        # Tombol
        button_layout = QHBoxLayout()
        
        cancel_button = QPushButton("Batal")
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
        
        delete_button = QPushButton("Hapus")
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

# Modifikasi kelas CameraItem untuk menampilkan kamera dari database
class CameraItem(QFrame):
    """Reusable widget untuk menampilkan satu kamera dalam daftar"""
    
    # Signal untuk navigasi ke detail kamera
    camera_clicked = Signal(int)  # Emit camera_id saat diklik
    edit_clicked = Signal(int)    # Emit camera_id saat tombol edit diklik
    delete_clicked = Signal(int)  # Emit camera_id saat tombol delete diklik
    
    def __init__(self, camera_id, camera_name, ip_address, port, protocol='RTSP',
                 username='', password='', stream_path='', url='',
                 is_online=False, preview_image=None, parent=None, is_static=False):
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
        self.is_static = is_static  
        # Konfigurasi frame
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
        
        # Buat widget bisa diklik
        self.setCursor(Qt.PointingHandCursor)
        self.mousePressEvent = self.on_click
        
        # Layout utama untuk item kamera
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Widget preview dengan rasio 16:9
        self.preview_widget = QLabel()
        self.preview_widget.setFixedSize(160, 90)  # Rasio 16:9
        self.preview_widget.setAlignment(Qt.AlignCenter)
        self.preview_widget.setStyleSheet("background-color: #1C1C1F; border-radius: 4px;")
        self.preview_widget.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        
        # Set gambar preview jika ada
        if preview_image:
            self.set_preview(preview_image)
        else:
            # Preview text
            preview_text = "Offline" if not is_online else "Live"
            self.preview_widget.setText(preview_text)
            self.preview_widget.setStyleSheet(f"""
                background-color: #1C1C1F; 
                color: {'#7f8c8d' if not is_online else '#4CAF50'}; 
                font-size: 14px;
                border-radius: 4px;
            """)
        
        # Tambahkan preview ke layout utama
        layout.addWidget(self.preview_widget)
        layout.addSpacing(10)
        
        # Bagian informasi kamera
        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)
        
        name_label = QLabel(camera_name)
        name_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #E4E4E7; border: 0px;")
        name_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        
        # Tampilkan RTSP URL
        url_display = self.url if self.url else "[No URL Set]"
        if len(url_display) > 40:
            # Truncate long URLs
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
        
        # Tambahkan bagian info ke layout utama
        layout.addLayout(info_layout, stretch=1)
        
        # Bagian tombol
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)
        buttons_layout.setAlignment(Qt.AlignVCenter)
        
        # Container tombol untuk alignment yang tepat
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
        if self.is_static:
            edit_button.setEnabled(False)
            delete_button.setEnabled(False)
            edit_button.setToolTip("Cannot edit a static camera")
            delete_button.setToolTip("Cannot delete a static camera")
            edit_button.setStyleSheet("background-color: #3F3F46; color: #71717A;")
            delete_button.setStyleSheet("background-color: #3F3F46; color: #71717A;")
        
        buttons_inner_layout.addWidget(edit_button)
        buttons_inner_layout.addStretch()
        buttons_inner_layout.addWidget(delete_button)
        
        buttons_layout.addWidget(buttons_container)
        buttons_layout.addStretch()
        
        # Tambahkan bagian tombol ke layout utama
        layout.addLayout(buttons_layout)
        
        self._status_worker: CameraStatusWorker | None = None

        if self.is_static:
            self.update_status(True)
    
    def set_preview(self, image):
        """
        Set gambar preview kamera.
        
        Args:
            image: QPixmap, atau numpy array (BGR format)
        """
        if isinstance(image, QPixmap):
            pixmap = image
        elif isinstance(image, np.ndarray):
            # AMAN: Konversi di GUI thread
            pixmap = convert_cv_to_pixmap(image, QSize(160, 90))
        else:
            return
        
        if not pixmap.isNull():
            self.preview_widget.setPixmap(pixmap)
            self.preview_widget.setStyleSheet("background-color: #1C1C1F; border-radius: 4px;")

    def _update_preview_label(self, frame):
        """Slot: terima frame dari CameraThread & tampilkan di QLabel preview."""
        pixmap = convert_cv_to_pixmap(frame, QSize(160, 90))
        if not pixmap.isNull():
            self.preview_widget.setPixmap(pixmap)
            # hapus teks/offline‑bg jika masih ada
            self.preview_widget.setStyleSheet("border-radius: 4px;")

    def update_status(self, is_online: bool):

        if self.is_static:
            self.is_online = True
            self.status_label.setText("Status: Online")
            self.status_label.setStyleSheet(f"font-size: 14px; color: #4CAF50; border: 0px;")
            return

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
        """Handler untuk klik pada item kamera"""
        # Emit sinyal dengan ID kamera
        self.camera_clicked.emit(self.camera_id)
    
    def on_edit_clicked(self):
        """Handler untuk tombol edit"""
        # Emit sinyal dengan ID kamera
        self.edit_clicked.emit(self.camera_id)
    
    def on_delete_clicked(self):
        """Handler untuk tombol delete"""
        # Emit sinyal dengan ID kamera
        self.delete_clicked.emit(self.camera_id)

    def closeEvent(self, event):

        super().closeEvent(event)


class CameraList(QWidget):
    """Widget kontainer yang menampilkan daftar kamera"""
    
    # Signal untuk navigasi ke detail kamera
    open_camera_detail = Signal(int)  # Emit camera_id untuk dibuka

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
        
        # Layout utama
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(0)
        self.setStyleSheet("""
            QWidget {
                border: 1px solid #27272A;
            }
        """)
        
        # Header
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
        
        # Tambahkan header ke layout utama
        self.main_layout.addWidget(header_frame)
        
        # Container untuk item kamera
        self.cameras_container = QWidget()
        self.cameras_layout = QVBoxLayout(self.cameras_container)
        self.cameras_layout.setContentsMargins(0, 0, 0, 0)
        self.cameras_layout.setAlignment(Qt.AlignTop)
        self.cameras_layout.setSpacing(0)
        
        # Scroll area
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
        
        # Tambahkan scroll area ke layout utama
        self.main_layout.addWidget(scroll_area)
        
        # Empty state untuk jika tidak ada kamera
        self.empty_label = QLabel("No cameras found. Click 'Add Camera' to get started.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.cameras_layout.addWidget(self.empty_label)
        self.empty_label.setVisible(True)
        
        # Muat daftar kamera dari database
        self.load_cameras()
    
    def load_cameras(self):
        """Memuat daftar kamera dari database secara asinkron."""
        # Membersihkan layout terlebih dahulu (logika Anda tetap sama)
        self.ping_pool.clear()

        while self.cameras_layout.count():
            item = self.cameras_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Logika untuk menampilkan pesan "No cameras found" tetap sama
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

        # PEMANGGILAN BENAR: Gunakan worker untuk get_all_cameras
        worker = DBWorker(signals, "get_all_cameras")
        QThreadPool.globalInstance().start(worker)

    # --- METHOD BARU YANG HILANG ---
    def _load_static_cameras(self):
        """Membaca file static_cameras.json dan mengembalikan list of dict."""
        static_cameras = []
        try:
            static_path = resource_path("app/assets/static_cameras.json")
            if os.path.exists(static_path):
                with open(static_path, 'r', encoding='utf-8') as f:
                    static_cameras = json.load(f)
        except Exception as e:
            logger.error(f"Gagal memuat kamera statis: {e}")
        return static_cameras

    # --- METHOD BARU YANG HILANG ---
    def _get_static_preview(self, video_path):
        """Mendapatkan frame pertama dari video sebagai preview."""
        try:
            # Pastikan path menggunakan separator yang benar untuk OS
            video_path = os.path.normpath(video_path)
            if not os.path.exists(video_path):
                logger.warning(f"Static video file not found at: {video_path}")
                return None
            
            cap = cv.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return frame
        except Exception as e:
            logger.error(f"Gagal mendapatkan preview dari {video_path}: {e}")
        return None

    def _populate_camera_list(self, cameras):
        """Populate camera list dan trigger initial snapshots"""

        self._hide_empty_label()

        # Gabungkan kamera dari DB dan file statis
        all_cameras = cameras + self._load_static_cameras()

        if not all_cameras:
            self._show_empty_state()
            return

        for camera_data in all_cameras:
            is_static = camera_data.get('is_static', False)
            preview = self.preview_cache.get(camera_data['id'], None)
            if is_static and preview is None:
                preview = self._get_static_preview(camera_data.get('video_path'))
                if preview is not None:
                    self.preview_cache[camera_data['id']] = convert_cv_to_pixmap(preview, QSize(160, 90))
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
                is_online=is_static,
                preview_image=preview if preview is not None else None,
                is_static=is_static,
            )
            camera_item.camera_clicked.connect(self.open_camera_detail)
            camera_item.edit_clicked.connect(self.edit_camera)
            camera_item.delete_clicked.connect(self.delete_camera)
            self.cameras_layout.addWidget(camera_item)

            # Store camera instance
            self.active_cameras[camera_data['id']] = Camera.from_dict(camera_data)
            
            if not is_static:
                self._request_initial_snapshot(camera_data['id'])

    def _request_initial_snapshot(self, camera_id: int):
        """Request snapshot SEKALI untuk camera saat pertama kali dimuat"""
        if camera_id in self.snapshot_done:
            return  # Sudah pernah di-snapshot
        
        # Mark as done (bahkan sebelum berhasil, untuk mencegah retry)
        self.snapshot_done.add(camera_id)
        
        # Request snapshot
        self.preview_scheduler.request_snapshot(
            camera_id,
            callback=self._on_snapshot_received,
        )

    def _hide_empty_label(self):
        """Helper method untuk menyembunyikan empty label dengan safe"""
        try:
            if (hasattr(self, "empty_label") and 
                self.empty_label is not None and 
                isValid(self.empty_label)):
                self.empty_label.setVisible(False)
        except RuntimeError:
            # C++ object sudah dihapus, tidak apa-apa
            pass
    
    def _show_empty_state(self):
        """
        Tampilkan label “No cameras …” — buat ulang jika label sudah ter‑delete.
        """
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
        """Menambahkan kamera baru ke database secara asinkron."""
        # Pola baru yang aman untuk worker
        signals = DBSignals()
        signals.finished.connect(self._get_added_camera_data)
        signals.error.connect(self.window()._db_error_msg)

        # Buat worker dengan memberikan objek sinyal
        worker = DBWorker(
            signals,
            "add_camera",
            name, ip_address, port, protocol, username, password,
            stream_path, url, roi_points=roi_points
        )
        
        # Jalankan di thread pool global
        QThreadPool.globalInstance().start(worker)
        return True

    @Slot(int)
    def _get_added_camera_data(self, camera_id: int):
        """Langkah 2: Mengambil data lengkap dari kamera yang baru ditambahkan."""
        if not camera_id:
            QMessageBox.warning(self, "Database Error", "Gagal mendapatkan ID untuk kamera baru.")
            return

        signals = DBSignals()
        # Hubungkan sinyal 'finished' (yang akan membawa data kamera lengkap) ke fungsi final
        signals.finished.connect(self._on_camera_added)
        signals.error.connect(self.window()._db_error_msg)

        # Buat worker untuk mengambil data kamera dari DB
        worker = DBWorker(signals, "get_camera", camera_id)
        QThreadPool.globalInstance().start(worker)
    
    @Slot(dict)
    def _on_camera_added(self, camera_data: dict):
        """Handler ketika camera baru ditambahkan"""
        if not camera_data:
            QMessageBox.warning(self, "Database Error", "Gagal mengambil data kamera yang baru ditambahkan.")
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

        # Request ONE-TIME snapshot untuk camera baru
        self._request_initial_snapshot(camera_id)
    
    def edit_camera(self, camera_id):
        """Langkah 1: Mengambil data kamera yang akan diedit secara asinkron."""
        signals = DBSignals()
        signals.finished.connect(self._on_edit_camera_data_loaded)
        signals.error.connect(self.window()._db_error_msg)

        # PEMANGGILAN BENAR: Gunakan worker untuk mengambil data sebelum membuka dialog edit
        worker = DBWorker(signals, "get_camera", camera_id)
        QThreadPool.globalInstance().start(worker)

    @Slot(dict)
    def _on_edit_camera_data_loaded(self, camera_data):
        """Langkah 2: Membuka dialog edit setelah data kamera berhasil diambil."""
        if not camera_data:
            QMessageBox.warning(self, "Error", "Camera data not found.")
            return

        dialog = AddCameraDialog(self.window(), camera_data)
        if not dialog.exec():
            return

        updated = dialog.get_camera_data()
        
        # Langkah 3: Menjalankan worker untuk memperbarui data di database
        update_signals = DBSignals()
        update_signals.finished.connect(lambda ok: self._on_camera_updated(ok, updated))
        update_signals.error.connect(self.window()._db_error_msg)

        # PEMANGGILAN BENAR: Gunakan worker untuk update_camera
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
        """Langkah 1: Mengambil data kamera untuk ditampilkan di dialog konfirmasi."""
        signals = DBSignals()
        signals.finished.connect(self._on_delete_camera_data_loaded)
        signals.error.connect(self.window()._db_error_msg)
        
        # PEMANGGILAN BENAR: Gunakan worker untuk mengambil data sebelum konfirmasi hapus
        worker = DBWorker(signals, "get_camera", camera_id)
        QThreadPool.globalInstance().start(worker)

    @Slot(dict)
    def _on_delete_camera_data_loaded(self, camera_data):
        """Langkah 2: Menampilkan dialog konfirmasi setelah data kamera didapat."""
        if not camera_data: return

        dialog = DeleteCameraDialog(camera_data['name'], self.window())
        if not dialog.exec(): return
            
        # Langkah 3: Menjalankan worker untuk menghapus kamera
        delete_signals = DBSignals()
        delete_signals.finished.connect(lambda ok: self._on_camera_deleted(ok, camera_data['name'], camera_data['id']))
        delete_signals.error.connect(self.error_handler)
        
        # PEMANGGILAN BENAR: Gunakan worker untuk delete_camera
        worker = DBWorker(delete_signals, "delete_camera", camera_data['id'])
        QThreadPool.globalInstance().start(worker)

    
    def _on_camera_deleted(self, success, camera_name, camera_id):
        """Handle completion of camera deletion."""
        if success:
            # Remove from caches
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
        
    def _refresh_previews(self):
        """
        Optional: Update preview HANYA untuk cameras yang sedang streaming.
        Method ini bisa dipanggil manual jika diperlukan.
        """
        for i in range(self.cameras_layout.count()):
            widget = self.cameras_layout.itemAt(i).widget()
            if not isinstance(widget, CameraItem):
                continue
                
            # HANYA update camera yang sedang streaming
            cam = self.active_cameras.get(widget.camera_id)
            if cam and cam.connection_status:
                frame = cam.get_last_frame()
                if frame is not None:
                    widget.set_preview(frame)
    
    def _on_snapshot_received(self, camera_id: int, frame: np.ndarray):
        """Callback di GUI thread saat snapshot diterima"""
        # Find the camera widget
        found = False
        for i in range(self.cameras_layout.count()):
            widget = self.cameras_layout.itemAt(i).widget()
            if isinstance(widget, CameraItem) and widget.camera_id == camera_id:
                # Convert numpy array ke QPixmap di GUI thread
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
        """Update status untuk satu CameraItem jika masih valid."""
        for i in range(self.cameras_layout.count()):
            widget = self.cameras_layout.itemAt(i).widget()
            if isinstance(widget, CameraItem) and widget.camera_id == camera_id:
                if isValid(widget):
                    widget.update_status(is_online)
                break

    def _refresh_statuses(self):
        """Periodic ping untuk update status online/offline"""
        for i in range(self.cameras_layout.count()):
            widget = self.cameras_layout.itemAt(i).widget()
            if not isinstance(widget, CameraItem):
                continue
            
            # Untuk camera yang sedang streaming, ambil frame terakhir
            cam = self.active_cameras.get(widget.camera_id)
            if cam and cam.connection_status:
                frame = cam.get_last_frame()
                if frame is not None:
                    widget.set_preview(frame)
                    widget.update_status(True)
                continue
            
            # Ping untuk update status
            worker = PingWorker(widget.camera_id, widget.ip_address, widget.port)
            worker.signals.finished.connect(self._update_item_status)
            self.ping_pool.start(worker)
    
    def showEvent(self, e):
        """Saat widget ditampilkan"""
        super().showEvent(e)
        # Hanya start ping timer
        if not self.ping_timer.isActive():
            self.ping_timer.start()
    
    def hideEvent(self, e):
        """Saat widget disembunyikan"""
        super().hideEvent(e)
        self.ping_timer.stop()

    def closeEvent(self, event):
        """Proper cleanup on close"""
        # Stop all timers
        self.ping_timer.stop()

        if hasattr(self, 'preview_scheduler'):
            self.preview_scheduler.cancel_all()
        
        # Wait for thread pools to finish
        self.db_pool.waitForDone(2000)
        self.ping_pool.waitForDone(2000)
        self.preview_pool.waitForDone(2000)
        QThreadPool.globalInstance().waitForDone(2000)
        
        # Release video capture pool
        from .utils.preview_scheduler import _capture_pool
        _capture_pool.release_all()
        
        super().closeEvent(event)


class MainWindow(QMainWindow):
    """Main window aplikasi"""
    
    def __init__(self):
        super().__init__()
        # Setup UI
        self.setup_ui()
        
        # Setup timer untuk jam
        self.update_clock()
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)  # Update setiap detik
        
        # Hitung kamera yang terhubung
        self.update_status_bar()
    
    def setup_ui(self):
        """Setup UI untuk main window"""
        # Main window setup
        self.setWindowTitle("EyeLog - Camera Monitoring")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #09090B;")
        self.setWindowIcon(QIcon(resource_path("assets/icons/pdu.png")))

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header frame
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.Panel)
        header_frame.setFixedHeight(57)
        header_frame.setStyleSheet("background-color: #EA580C; border: 0px;")
        
        # Header layout
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 0, 20, 0)

        # Title icon and text for header
        header_icon_label = QLabel()
        icon_pixmap = QIcon(resource_path("assets/icons/pdu.png")).pixmap(QSize(30, 30))
        if not icon_pixmap.isNull():
            header_icon_label.setPixmap(icon_pixmap)
        else:
            # Fallback jika icon tidak ditemukan
            header_icon_label.setText("PDU")
            header_icon_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        
        header_title_label = QLabel("PDU | Parama Data Unit")
        header_title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        header_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Clock widgets icon and text
        clock_icon_label = QLabel()
        clock_icon_pixmap = QIcon(resource_path("assets/icons/clock.png")).pixmap(QSize(18, 18))
        if not clock_icon_pixmap.isNull():
            clock_icon_label.setPixmap(clock_icon_pixmap)
        
        self.clock_text_label = QLabel()
        self.clock_text_label.setStyleSheet("font-size: 14px; color: #FFFFFF;")

        # Notification button icon
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
        
        # Add widgets to header
        header_layout.addWidget(header_icon_label)
        header_layout.addSpacing(4)
        header_layout.addWidget(header_title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(clock_icon_label)
        header_layout.addSpacing(3)
        header_layout.addWidget(self.clock_text_label)
        header_layout.addStretch(1)
        header_layout.addWidget(notification_button)

        # Navbar frame
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
        
        # Camera list widget
        self.camera_list = CameraList(error_handler=self._db_error_msg, parent=self)
        self.camera_list.open_camera_detail.connect(self.open_camera_detail)
        
        # Add widgets to main layout
        main_layout.addWidget(header_frame)
        main_layout.addWidget(navbar_frame)
        main_layout.addWidget(self.camera_list)
        
        # Status bar
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #27272A;
                color: #E4E4E7;
            }
        """)
        self.showMaximized()
    
    def update_clock(self):
        """Update jam pada header"""
        current_time = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
        self.clock_text_label.setText(current_time)
    
    def show_notification_dialog(self):
        """
        Tampilkan dialog modal yang menandakan fitur belum tersedia.
        QMessageBox.exec() membuat dialog ini modal sehingga
        jendela utama tidak bisa dioperasikan sebelum ditekan OK.
        """
        QMessageBox.information(
            self,
            "Coming Soon",
            "This feature is under construction / not implemented yet.",
            QMessageBox.Ok
        )

    def update_status_bar(self):
        """Memperbarui status bar dengan jumlah kamera secara asinkron."""
        signals = DBSignals()
        signals.finished.connect(lambda cameras: self.statusBar().showMessage(f"{len(cameras)} Cameras Connected | Database: Connected"))
        signals.error.connect(self._db_error_msg)
        
        worker = DBWorker(signals, "get_all_cameras")
        QThreadPool.globalInstance().start(worker)

    def pause_background_tasks(self):
        """Pause all background tasks when camera detail is opened"""
        # Stop clock timer
        if hasattr(self, 'clock_timer') and self.clock_timer.isActive():
            self.clock_timer.stop()
        
        # Stop ping timer in camera list
        if hasattr(self, 'camera_list') and hasattr(self.camera_list, 'ping_timer'):
            if self.camera_list.ping_timer.isActive():
                self.camera_list.ping_timer.stop()
        
        # Optional: Clear pending workers to free thread pool
        if hasattr(self.camera_list, 'ping_pool'):
            self.camera_list.ping_pool.clear()

    def resume_background_tasks(self):
        """Resume all background tasks when returning from camera detail"""
        # Restart clock timer
        if hasattr(self, 'clock_timer'):
            self.clock_timer.start(1000)
            self.update_clock()  # Update immediately
        
        # Restart ping timer in camera list
        if hasattr(self, 'camera_list') and hasattr(self.camera_list, 'ping_timer'):
            self.camera_list.ping_timer.start(30000)
    
    def show_add_camera_dialog(self):
        """Tampilkan dialog untuk menambahkan kamera baru"""
        dialog = AddCameraDialog(self)
        
        if dialog.exec():
            camera_data = dialog.get_camera_data()
            # Panggil metode di CameraList untuk menangani penambahan
            self.camera_list.add_camera(
                camera_data['name'], camera_data['ip_address'], camera_data['port'],
                camera_data['protocol'], camera_data['username'], camera_data['password'],
                camera_data['stream_path'], camera_data['url'],
                roi_points=camera_data.get('roi_points')
            )
    
    def open_camera_detail(self, camera_id):
        """
        Membuka halaman detail kamera.
        - Untuk kamera statis (ID < 0), buka secara langsung.
        - Untuk kamera DB (ID > 0), ambil data secara asinkron.
        """
        # --- PATH 1: Untuk kamera statis (ID negatif, sinkron) ---
        if camera_id < 0:
            camera_obj = self.camera_list.active_cameras.get(camera_id)
            if not camera_obj:
                QMessageBox.warning(self, "Camera Not Found", f"Static camera with ID {camera_id} not found.")
                return

            self.pause_background_tasks()
            
            # Buka view statis yang disederhanakan
            detail_window = CameraDetailUI_Static(camera_obj.__dict__, parent=self)
            detail_window.destroyed.connect(self.resume_background_tasks)
            self.hide()
            detail_window.show()

        # --- PATH 2: Untuk kamera dari database (ID positif, asinkron seperti asli) ---
        else:
            # [cite_start]Menggunakan pola worker dan sinyal seperti kode asli [cite: 146, 149]
            signals = DBSignals()
            
            @Slot(dict)
            def _on_data_received(camera_data):
                if camera_data:
                    self.pause_background_tasks()
                    
                    # Buka view normal yang memiliki fungsionalitas penuh
                    detail_window = CameraDetailUI(camera_data, parent=self)
                    detail_window.destroyed.connect(self.resume_background_tasks)
                    
                    self.hide()
                    detail_window.show()
                else:
                    QMessageBox.warning(self, "Camera Not Found", f"Camera with ID {camera_id} not found in database.")

            signals.finished.connect(_on_data_received)
            signals.error.connect(self._db_error_msg)
            
            # Worker memanggil "get_camera" dari DB untuk ID positif
            worker = DBWorker(signals, "get_camera", camera_id)
            QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _db_error_msg(self, error_message: str):
        """Menampilkan pesan error dari database secara terpusat."""
        logger.error(f"Database operation failed: {error_message}")
        QMessageBox.critical(self, "Database Error", error_message)
    
    def closeEvent(self, event):
        """Cleanup saat aplikasi ditutup"""
        # Stop all timers
        if hasattr(self, 'clock_timer'):
            self.clock_timer.stop()
        
        # Cleanup camera list
        if hasattr(self, 'camera_list'):
            self.camera_list.close()
        
        # Accept the close event
        super().closeEvent(event)
    


if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec())
