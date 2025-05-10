import sys
import os
import json
import time
import cv2 as cv
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QFrame, QScrollArea, QSizePolicy, 
                              QMessageBox, QDialog)
from PySide6.QtCore import Qt, QSize, QTimer, Signal, QObject, QThread
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap

from resources import resource_path
from models.camera import Camera, convert_cv_to_pixmap, CameraThread
from views.add_camera import AddCameraDialog, validate_ip_address
from models.database import DatabaseManager


# Komponen untuk dialog konfirmasi hapus kamera
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

# Worker untuk mendeteksi status kamera secara asinkron
class CameraStatusWorker(QObject):
    """Worker untuk memeriksa status kamera di thread terpisah"""
    status_changed = Signal(bool)
    finished = Signal()
    
    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port
    
    def check_status(self):
        """Periksa status kamera"""
        is_online = validate_ip_address(self.ip, self.port)
        self.status_changed.emit(is_online)
        self.finished.emit()

# Modifikasi kelas CameraItem untuk menampilkan kamera dari database
class CameraItem(QFrame):
    """Reusable widget untuk menampilkan satu kamera dalam daftar"""
    
    # Signal untuk navigasi ke detail kamera
    camera_clicked = Signal(int)  # Emit camera_id saat diklik
    edit_clicked = Signal(int)    # Emit camera_id saat tombol edit diklik
    delete_clicked = Signal(int)  # Emit camera_id saat tombol delete diklik
    
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
        self.preview_stream: Camera | None = None
        
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
        
        # Tampilkan RTSP URL
        url_display = self.url if self.url else "[No URL Set]"
        if len(url_display) > 40:
            # Truncate long URLs
            url_display = url_display[:37] + "..."
        
        ip_label = QLabel(f"URL: {url_display}")
        ip_label.setStyleSheet("font-size: 14px; color: #E4E4E7; border: 0px;")
        
        status_color = "#4CAF50" if is_online else "#9CA3AF"
        status_text = "Online" if is_online else "Offline"
        self.status_label = QLabel(f"Status: {status_text}")
        self.status_label.setStyleSheet(f"font-size: 14px; color: {status_color}; border: 0px;")
        
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
        
        buttons_inner_layout.addWidget(edit_button)
        buttons_inner_layout.addStretch()
        buttons_inner_layout.addWidget(delete_button)
        
        buttons_layout.addWidget(buttons_container)
        buttons_layout.addStretch()
        
        # Tambahkan bagian tombol ke layout utama
        layout.addLayout(buttons_layout)
        
        # Timer untuk memeriksa status kamera
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.check_camera_status)
        self.status_timer.start(30000)  # Periksa setiap 30 detik

        self._status_thread: QThread | None = None
        self._status_worker: CameraStatusWorker | None = None
        
        # Lakukan pemeriksaan status awal
        QTimer.singleShot(100, self.check_camera_status)
    
    def set_preview(self, image):
        """Set gambar preview kamera"""
        if isinstance(image, QPixmap):
            pixmap = image
        else:
            # Konversi dari OpenCV image ke QPixmap
            pixmap = convert_cv_to_pixmap(image, QSize(160, 90))
        
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
    
    def get_preview_frame(self):
        """Get a single frame from the camera stream for preview"""
        try:
        #  ➜ 1. Bangun URL sekali saja
            rtsp_url = Camera(
                name=self.camera_name, ip_address=self.ip_address,
                port=self.port, username=self.username,
                password=self.password, stream_path=self.stream_path,
                custom_url=self.url).build_stream_url()

            #  ➜ 2. Buka langsung dengan OpenCV, ambil 1 frame, lalu tutup
            cap = cv.VideoCapture(rtsp_url)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 160)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 90)
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                pixmap = convert_cv_to_pixmap(frame, QSize(160, 90))
                self.preview_widget.setPixmap(pixmap)
        except Exception as e:
            print(f"[Preview] {e}")

    def update_status(self, is_online):
        self.is_online = is_online
        status_color = "#4CAF50" if is_online else "#9CA3AF"
        status_text = "Online" if is_online else "Offline"
        self.status_label.setText(f"Status: {status_text}")
        self.status_label.setStyleSheet(f"font-size: 14px; color: {status_color}; border: 0px;")

        if is_online:
            # --- guard: jika sudah ada stream, tidak buat ulang ---
            if self.preview_stream:
                return
            cam = Camera(name=self.camera_name,
                        ip_address=self.ip_address,
                        port=self.port,
                        username=self.username,
                        password=self.password,
                        stream_path=self.stream_path,
                        custom_url=self.url)
            cam.is_preview_mode = True
            if cam.connect():
                cam.start_stream()
                if cam.thread:
                    cam.thread.frame_received.connect(self._update_preview_label)
                self.preview_stream = cam

        else:  # offline
            if self.preview_stream:
                self.preview_stream.disconnect()
                self.preview_stream = None
            self.preview_widget.clear()
            self.preview_widget.setText("Offline")
            self.preview_widget.setStyleSheet(
                "background-color: #1C1C1F; color: #7f8c8d; font-size: 14px; border-radius: 4px;"
            )

    def check_camera_status(self):
        if not self.isVisible():
            return
        
        if self._status_thread and self._status_thread.isRunning():
            return
        
        self._status_thread = None

        self._status_thread = QThread(self)
        self._status_worker = CameraStatusWorker(self.ip_address, self.port)
        self._status_worker.moveToThread(self._status_thread)

        self._status_thread.started.connect(self._status_worker.check_status)
        self._status_worker.status_changed.connect(self.update_status)

        #  Pastikan urutan quit/delete benar
        self._status_worker.finished.connect(self._status_thread.quit)
        self._status_worker.finished.connect(self._status_worker.deleteLater)
        self._status_thread.finished.connect(self._status_thread.deleteLater)

        self._status_thread.finished.connect(
            lambda: setattr(self, "_status_thread", None)
        )

        self._status_thread.finished.connect(
            lambda: setattr(self, "_status_worker", None)
        )

        self._status_thread.start()
    
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
        if self._status_thread and self._status_thread.isRunning():
            self._status_thread.quit()
            self._status_thread.wait()

        if self.preview_stream:
            self.preview_stream.disconnect()
            self.preview_stream = None

        super().closeEvent(event)


class CameraList(QWidget):
    """Widget kontainer yang menampilkan daftar kamera"""
    
    # Signal untuk navigasi ke detail kamera
    open_camera_detail = Signal(int)  # Emit camera_id untuk dibuka
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        
        # Database manager
        self.db_manager = db_manager
        
        # Dictionary untuk menyimpan instance kamera aktif
        self.active_cameras = {}
        
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
        """Memuat daftar kamera dari database"""
        # Bersihkan layout terlebih dahulu
        while self.cameras_layout.count() > 0:
            item = self.cameras_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Dapatkan semua kamera dari database
        cameras = self.db_manager.get_all_cameras()
        
        # Tampilkan pesan kosong jika tidak ada kamera
        if not cameras:
            self.empty_label = QLabel("No cameras found. Click 'Add Camera' to get started.")
            self.empty_label.setAlignment(Qt.AlignCenter)
            self.empty_label.setStyleSheet("color: #71717A; font-size: 16px; padding: 40px;")
            self.cameras_layout.addWidget(self.empty_label)
            return
        
        # Tambahkan setiap kamera ke layout
        for camera_data in cameras:
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
                is_online=False  # Default offline, akan diperiksa oleh item
            )
            
            # Hubungkan sinyal
            camera_item.camera_clicked.connect(self.open_camera_detail)
            camera_item.edit_clicked.connect(self.edit_camera)
            camera_item.delete_clicked.connect(self.delete_camera)
            
            self.cameras_layout.addWidget(camera_item)
    
    def add_camera(self, name, ip_address, port, protocol="RTSP", username="", password="", stream_path="", url="", roi_points=None):
        """Menambahkan kamera baru ke daftar"""
        # Tambahkan ke database
        camera_id = self.db_manager.add_camera(
            name, ip_address, port, protocol, 
            username, password, stream_path, url,
            roi_points=roi_points
        )
        
        if camera_id:
            # Hapus pesan empty state jika ada
            if self.empty_label and self.empty_label.isVisible():
                self.empty_label.setVisible(False)
                self.empty_label.deleteLater()
                self.empty_label = None
            
            # Buat item kamera baru
            camera_item = CameraItem(
                camera_id=camera_id,
                camera_name=name,
                ip_address=ip_address,
                port=port,
                protocol=protocol,
                username=username,
                password=password,
                stream_path=stream_path,
                url=url,
            )
            
            # Hubungkan sinyal
            camera_item.camera_clicked.connect(self.open_camera_detail)
            camera_item.edit_clicked.connect(self.edit_camera)
            camera_item.delete_clicked.connect(self.delete_camera)
            
            # Tambahkan ke layout
            self.cameras_layout.addWidget(camera_item)
            
            return camera_id
        
        return None
    
    def edit_camera(self, camera_id):
        """Handler untuk mengedit kamera"""
        # Ambil data kamera
        camera_data = self.db_manager.get_camera(camera_id)
        
        if camera_data:
            # Buka dialog edit
            dialog = AddCameraDialog(self.window(), camera_data)
            
            if dialog.exec():
                # Update data kamera
                updated_data = dialog.get_camera_data()
                success = self.db_manager.update_camera(
                    camera_id,
                    updated_data['name'],
                    updated_data['ip_address'],
                    updated_data['port'],
                    updated_data['protocol'],
                    updated_data['username'],
                    updated_data['password'],
                    updated_data['stream_path'],
                    updated_data['url']
                )
                
                if success:
                    # Reload daftar kamera
                    self.load_cameras()
                    
                    # Tampilkan pesan sukses
                    QMessageBox.information(
                        self.window(),
                        "Camera Updated",
                        f"Camera '{updated_data['name']}' has been updated successfully."
                    )
                else:
                    QMessageBox.warning(
                        self.window(),
                        "Update Failed",
                        "Failed to update camera. Please try again."
                    )
    
    def delete_camera(self, camera_id):
        """Handler untuk menghapus kamera"""
        # Ambil data kamera
        camera_data = self.db_manager.get_camera(camera_id)
        
        if camera_data:
            # Tampilkan dialog konfirmasi
            dialog = DeleteCameraDialog(camera_data['name'], self.window())
            
            if dialog.exec():
                # Hapus kamera dari database
                success = self.db_manager.delete_camera(camera_id)
                
                if success:
                    # Reload daftar kamera
                    self.load_cameras()
                    
                    # Tampilkan pesan sukses
                    QMessageBox.information(
                        self.window(),
                        "Camera Deleted",
                        f"Camera '{camera_data['name']}' has been deleted."
                    )
                else:
                    QMessageBox.warning(
                        self.window(),
                        "Delete Failed",
                        "Failed to delete camera. Please try again."
                    )


class MainWindow(QMainWindow):
    """Main window aplikasi"""
    
    def __init__(self):
        super().__init__()
        
        # Inisialisasi database manager
        self.db_manager = DatabaseManager()
        
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
        self.showMaximized()

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
        self.camera_list = CameraList(self.db_manager)
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
        """Update status bar dengan jumlah kamera terhubung"""
        cameras = self.db_manager.get_all_cameras()
        camera_count = len(cameras)
        self.statusBar().showMessage(f"{camera_count} Cameras Connected | Database: Connected")
    
    def show_add_camera_dialog(self):
        """Tampilkan dialog untuk menambahkan kamera baru"""
        dialog = AddCameraDialog(self)
        
        if dialog.exec():
            # Tambahkan kamera baru
            camera_data = dialog.get_camera_data()
            camera_id = self.camera_list.add_camera(
                camera_data['name'],
                camera_data['ip_address'],
                camera_data['port'],
                camera_data['protocol'],
                camera_data['username'],
                camera_data['password'],
                camera_data['stream_path'],
                camera_data['url'],
                roi_points=camera_data.get('roi_points'),
            )
            
            if camera_id:
                # Update status bar
                self.update_status_bar()
                
                # Tampilkan pesan sukses
                QMessageBox.information(
                    self,
                    "Camera Added",
                    f"Camera '{camera_data['name']}' has been added successfully."
                )
            else:
                QMessageBox.warning(
                    self,
                    "Add Failed",
                    "Failed to add camera. Please try again."
                )
    
    def open_camera_detail(self, camera_id):
        """Buka halaman detail kamera"""
        # Ambil data kamera dari database
        camera_data = self.db_manager.get_camera(camera_id)

        if camera_data:
            from views.camera_detail import CameraDetailUI
            camera_detail = CameraDetailUI(camera_data, parent=self)

            self.hide()
            self._detail_window = camera_detail

            camera_detail.init_camera()
            camera_detail.show()
        else:
            QMessageBox.warning(
                self,
                "Camera Not Found",
                f"Camera with ID {camera_id} not found."
            )

    def showEvent(self, event):
        """
        Dipanggil otomatis setiap kali MainWindow muncul kembali.
        Kembalikan semua preview ke FPS rendah (mode preview).
        """
        super().showEvent(event)

        # Iterasi semua item di CameraList
        layout = self.camera_list.cameras_layout
        for i in range(layout.count()):
            item = layout.itemAt(i).widget()
            if not item:
                continue

            cam = getattr(item, "preview_stream", None)  # Camera atau None
            if cam:
                cam.set_preview_mode(True) 


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
