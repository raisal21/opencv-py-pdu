import sys
import socket
import cv2 as cv
import time
import logging
from functools import partial 
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFormLayout, 
    QGroupBox, QSpinBox, QFrame, QSizePolicy,
    QMessageBox
)
from PySide6.QtCore import Qt, QSize, QRegularExpression
from PySide6.QtGui import QIcon, QFont, QRegularExpressionValidator
from ..resources import resource_path
from .views.roi_window import ROISelectorDialog 

logger = logging.getLogger(__name__)

def _refresh_styles(*widgets):
    """
    Unpolish/polish sekali untuk sekumpulan widget
    agar menghindari repaint berulang.
    """
    for w in widgets:
        w.style().unpolish(w)
        w.style().polish(w)

def validate_ip_address(ip_address, port, timeout=1):
    """Validasi alamat IP dengan mencoba koneksi"""
    try:
        # Cek format IP address
        socket.inet_aton(ip_address)
        
        # Coba untuk membuka koneksi
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip_address, port))
        sock.close()
        
        # Jika koneksi berhasil (result = 0)
        return result == 0
    except socket.error:
        return False


class AddCameraDialog(QDialog):
    def __init__(self, parent=None, camera_data=None):
        """
        Dialog untuk menambah atau mengedit kamera
        
        Args:
            parent: Parent widget
            camera_data: Dictionary berisi data kamera untuk mode edit
        """
        super().__init__(parent)
        
        # Simpan data kamera jika dalam mode edit
        self.edit_mode = camera_data is not None
        self.camera_data = camera_data or {}

        # Initialize ROI data
        self.roi_points = camera_data.get('roi_points') if self.edit_mode else None
        self.roi_image = None
        
        # Set dialog properties
        self.setWindowTitle("Edit Camera" if self.edit_mode else "Add Camera")
        self.setWindowIcon(QIcon(resource_path("assets/icons/webcam.png")))
        self.setMinimumSize(500, 400)  
        self.setStyleSheet("""
            QDialog {
                background-color: #09090B;
                border: 1px solid #27272A;
            }
            QLabel {
                color: #E4E4E7;
                border: 0px;
            }
            QGroupBox {
                color: #E4E4E7;
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #27272A;
                border-radius: 4px;
                margin-top: 12px;
                background-color: #09090B;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QSpinBox {
                background-color: #27272A;
                color: #E4E4E7;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
                selection-background-color: #EA580C;
            }
            QLineEdit:focus, QSpinBox:focus {
                border: 1px solid #EA580C;
            }
            QLineEdit[invalid="true"] {
                border: 1px solid #DC2626;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                border: 0px;
                background-color: #3F3F46;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #EA580C;
            }
        """)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create header frame
        header_frame = QFrame()
        header_frame.setFixedHeight(57)
        header_frame.setStyleSheet("background-color: #EA580C; border: 0px;")
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        # Header title
        header_title = QLabel("Edit Camera" if self.edit_mode else "Add Camera")
        header_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        
        # Close button
        close_button = QPushButton()
        close_button.setFixedSize(35, 35)
        close_button.setIcon(QIcon(resource_path("assets/icons/close.png")))
        close_button.setIconSize(QSize(16, 16))
        close_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #09090B;
            }
        """)
        close_button.clicked.connect(self.reject)
        
        header_layout.addWidget(header_title)
        header_layout.addStretch()
        header_layout.addWidget(close_button)
        
        # Add header to main layout
        main_layout.addWidget(header_frame)
        
        # Content area
        content_frame = QFrame()
        content_frame.setStyleSheet("border: 0px;")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)
        
        # Form group
        form_group = QGroupBox("Camera Configuration")
        form_group.setStyleSheet("""
            QGroupBox {
                margin-top: 20px;
                padding-top: 16px;
            }
        """)
        form_layout = QFormLayout(form_group)
        form_layout.setContentsMargins(20, 30, 20, 20)
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Form labels
        name_label = QLabel("Camera Name:")
        name_label.setStyleSheet("font-size: 12px;")
        ip_label = QLabel("Camera IP:")
        ip_label.setStyleSheet("font-size: 12px;")
        port_label = QLabel("Port:")
        port_label.setStyleSheet("font-size: 12px;")
        username_label = QLabel("Username (optional):")
        username_label.setStyleSheet("font-size: 12px;")
        password_label = QLabel("Password (optional):")
        password_label.setStyleSheet("font-size: 12px;")
        stream_label = QLabel("Stream Path:")
        stream_label.setStyleSheet("font-size: 12px;")
        
        # Form fields
        self.camera_name = QLineEdit()
        self.camera_name.setPlaceholderText("Enter camera name")
        self.camera_name.setMinimumHeight(36)
        self.camera_name.setProperty("invalid", False)
        
        # IP address validator
        ip_regex = QRegularExpression(
            "^(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\\."
            "(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\\."
            "(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\\."
            "(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)$"
        )
        ip_validator = QRegularExpressionValidator(ip_regex)
        
        self.camera_ip = QLineEdit()
        self.camera_ip.setPlaceholderText("Enter IP address (e.g. 192.168.1.100)")
        self.camera_ip.setMinimumHeight(36)
        self.camera_ip.setValidator(ip_validator)
        self.camera_ip.setProperty("invalid", False)
        
        self.camera_port = QSpinBox()
        self.camera_port.setRange(1, 65535)
        self.camera_port.setValue(554)  # Default RTSP port
        self.camera_port.setMinimumHeight(36)
        
        # Username field
        self.username = QLineEdit()
        self.username.setPlaceholderText("Username (e.g. admin)")
        self.username.setMinimumHeight(36)
        
        # Password field
        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setMinimumHeight(36)
        
        # Stream path field
        self.stream_path = QLineEdit()
        self.stream_path.setPlaceholderText("/stream1")
        self.stream_path.setMinimumHeight(36)
        
        # Error labels
        self.name_error = QLabel("")
        self.name_error.setStyleSheet("color: #DC2626; font-size: 10px;")
        self.ip_error = QLabel("")
        self.ip_error.setStyleSheet("color: #DC2626; font-size: 10px;")
        self.stream_error = QLabel("")
        self.stream_error.setStyleSheet("color: #DC2626; font-size: 10px;")
        
        # Add fields to form
        form_layout.addRow(name_label, self.camera_name)
        form_layout.addRow("", self.name_error)
        form_layout.addRow(ip_label, self.camera_ip)
        form_layout.addRow("", self.ip_error)
        form_layout.addRow(port_label, self.camera_port)
        form_layout.addRow(username_label, self.username)
        form_layout.addRow(password_label, self.password)
        form_layout.addRow(stream_label, self.stream_path)
        form_layout.addRow("", self.stream_error)
        
        # Populate fields if in edit mode
        if self.edit_mode:
            self.camera_name.setText(self.camera_data.get('name', ''))
            self.camera_ip.setText(self.camera_data.get('ip_address', ''))
            self.camera_port.setValue(self.camera_data.get('port', 554))
            self.username.setText(self.camera_data.get('username', ''))
            self.password.setText(self.camera_data.get('password', ''))
            self.stream_path.setText(self.camera_data.get('stream_path', ''))
        
        # Add form to content layout
        content_layout.addWidget(form_group)
        content_layout.addStretch()
        
        # Add content frame to main layout
        main_layout.addWidget(content_frame, 1)
        
        # Button area
        button_frame = QFrame()
        button_frame.setFixedHeight(70)
        button_frame.setStyleSheet("background-color: #27272A; border: 0px;")
        
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(20, 0, 20, 0)
        button_layout.setSpacing(10)
        
        # Create buttons
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedHeight(35)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #3F3F46;
                color: #E4E4E7;
                font-size: 14px;
                border-radius: 8px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #09090B;
                border: 1px solid #3F3F46;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        
        self.save_button = QPushButton("Save" if self.edit_mode else "Next")
        self.save_button.setFixedHeight(35)
        self.save_button.setStyleSheet("""
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
        self.save_button.setIcon(QIcon(resource_path("assets/icons/arrow-right.png")))
        self.save_button.setIconSize(QSize(18, 18))
        self.save_button.clicked.connect(self.validate_and_accept)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        
        # Add button frame to main layout
        main_layout.addWidget(button_frame)
    
    def validate_and_accept(self):
        """Validasi input lalu lanjut ke pemilihan ROI jika valid."""
        for f in (self.camera_name, self.camera_ip, self.stream_path):
            f.setProperty("invalid", False)
        self.name_error.setText("")
        self.ip_error.setText("")
        self.stream_error.setText("")
        _refresh_styles(self.camera_name, self.camera_ip, self.stream_path)   # <<< satu repaint

        camera_name = self.camera_name.text().strip()
        if not camera_name:
            self.camera_name.setProperty("invalid", True)
            self.name_error.setText("Camera name is required")
            _refresh_styles(self.camera_name)        # <<< repaint sekali saja untuk field ini
            return

        ip_address = self.camera_ip.text().strip()
        if not ip_address:
            self.camera_ip.setProperty("invalid", True)
            self.ip_error.setText("IP address is required")
            _refresh_styles(self.camera_ip)
            return

        stream_path = self.stream_path.text().strip()
        if not stream_path:
            self.stream_path.setProperty("invalid", True)
            self.stream_error.setText("Stream path is required")
            _refresh_styles(self.stream_path)
            return

        port = self.camera_port.value()
        wait_dialog = QMessageBox(self)
        wait_dialog.setWindowTitle("Validating Connection")
        wait_dialog.setText("Testing connection to camera...")
        wait_dialog.setStandardButtons(QMessageBox.NoButton)

        self.setEnabled(False)
        wait_dialog.show()
        QApplication.processEvents()

        is_valid = validate_ip_address(ip_address, port, timeout=1)

        wait_dialog.hide()
        QApplication.processEvents()
        self.setEnabled(True)

        if not is_valid:
            self.camera_ip.setProperty("invalid", True)
            self.ip_error.setText("Cannot connect to this IP address and port")
            _refresh_styles(self.camera_ip)
            return

        if self.edit_mode:
            self.accept()
        else:
            self.proceed_to_roi_selection()

    def proceed_to_roi_selection(self):
        """Lanjutkan ke langkah pemilihan ROI"""
        wait_dialog = QMessageBox(self)
        wait_dialog.setWindowTitle("Connecting to Camera")
        wait_dialog.setText("Connecting to camera stream for ROI selection...")
        wait_dialog.setStandardButtons(QMessageBox.NoButton)
        
        wait_dialog.show()
        QApplication.processEvents()
        
        try:
            # Build RTSP URL
            camera_data = self.get_camera_data()
            rtsp_url = camera_data['url']
            
            # Create VideoCapture
            cap = cv.VideoCapture(rtsp_url)
            
            # Set larger resolution for ROI selection
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Try to get a frame (with multiple attempts)
            frame = None
            max_attempts = 5
            
            for attempt in range(max_attempts):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                time.sleep(0.5)  # Wait a bit before next attempt
            
            # Close capture
            cap.release()
            
            # Close wait dialog
            wait_dialog.hide()
            QApplication.processEvents()
            
            if frame is None:
                QMessageBox.critical(
                    self,
                    "Connection Error",
                    "Could not get a video frame from the camera. Please check the camera settings."
                )
                return
            
            # Open ROI selector with the frame
            roi_dialog = ROISelectorDialog(frame, self)
            if roi_dialog.exec():
                # Get ROI data if user confirmed
                self.roi_points, self.roi_image = roi_dialog.get_roi()

                logger.debug("ROI Points obtained:", self.roi_points)
                logger.debug("ROI Points type:", type(self.roi_points))
                logger.debug("ROI Image shape:", self.roi_image.shape if self.roi_image is not None else None)
                
                # Dialog was accepted and ROI was selected, complete the camera addition
                self.accept()
            # If ROI dialog was rejected, stay on this dialog
            
        except Exception as e:
            wait_dialog.hide()
            QApplication.processEvents()
            
            QMessageBox.critical(
                self,
                "Connection Error",
                f"An error occurred while connecting to the camera: {str(e)}"
            )
    
    def get_camera_data(self):
        """Return the entered camera data"""
        username = self.username.text().strip()
        password = self.password.text().strip()
        ip = self.camera_ip.text().strip()
        port = self.camera_port.value()
        stream_path = self.stream_path.text().strip()
        
        # Ensure stream path starts with /
        if stream_path and not stream_path.startswith('/'):
            stream_path = '/' + stream_path
            
        # Build RTSP URL
        auth_part = f"{username}:{password}@" if username and password else ""
        url = f"rtsp://{auth_part}{ip}:{port}{stream_path}"
        
        # Create camera data dictionary
        camera_data = {
            'id': self.camera_data.get('id') if self.edit_mode else None,
            'name': self.camera_name.text().strip(),
            'ip_address': self.camera_ip.text().strip(),
            'port': self.camera_port.value(),
            'protocol': 'RTSP', 
            'username': username,
            'password': password,
            'stream_path': stream_path,
            'url': url  # Full URL for connection
        }
        
        # Add ROI data if available
        if self.roi_points:
            import json
            camera_data['roi_points'] = json.dumps(self.roi_points)
        
        return camera_data