import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QFrame, QScrollArea, QSizePolicy)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor, QPalette, QIcon

class CameraItem(QFrame):
    """Reusable widget untuk menampilkan satu kamera dalam daftar"""
    
    def __init__(self, camera_name="Camera Default", ip_address="192.168.1.100", is_online=True, parent=None):
        super().__init__(parent)
        
        # Konfigurasi frame
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QFrame {
                background-color: #09090B;
                border: 1px solid #27272A;
            }
        """)
        
        # Layout utama untuk item kamera
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Widget preview dengan rasio 16:9
        self.preview_widget = QWidget()
        self.preview_widget.setFixedSize(160, 90)  # Rasio 16:9
        
        # Set warna background berdasarkan status
        palette = self.preview_widget.palette() 
        bg_color = QColor("#27272A") if is_online else QColor("#E4E4E7")
        palette.setColor(QPalette.Window, bg_color)
        self.preview_widget.setAutoFillBackground(True)
        self.preview_widget.setPalette(palette)
        
        # Preview label
        preview_layout = QVBoxLayout(self.preview_widget)
        preview_layout.setAlignment(Qt.AlignCenter)
        
        if not is_online:
            status_label = QLabel("Offline")
            status_label.setStyleSheet("color: #7f8c8d; font-size: 12px;")
            preview_layout.addWidget(status_label, alignment=Qt.AlignCenter)
        
        preview_text = QLabel("Preview")
        preview_text.setStyleSheet("color: #7f8c8d; font-size: 10px; border: 0px;")
        preview_layout.addWidget(preview_text, alignment=Qt.AlignCenter)
        
        # Tambahkan preview ke layout utama
        layout.addWidget(self.preview_widget)
        layout.addSpacing(10)
        
        # Bagian informasi kamera
        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)
        
        name_label = QLabel(camera_name)
        name_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #E4E4E7; border: 0px;")
        
        ip_label = QLabel(f"IP Address: {ip_address}")
        ip_label.setStyleSheet("font-size: 14px; color: #E4E4E7; border: 0px;")
        
        status_color = "#EA580C" if is_online else "#7f8c8d"
        status_text = "Online" if is_online else "Offline"
        status_label = QLabel(f"Status: {status_text}")
        status_label.setStyleSheet(f"font-size: 14px; color: {status_color}; border: 0px;")
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(ip_label)
        info_layout.addWidget(status_label)
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
        edit_button.setIcon(QIcon("app/assets/icons/edit.png"))
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
        
        delete_button = QPushButton()
        delete_button.setIcon(QIcon("app/assets/icons/trash.png"))
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
        
        buttons_inner_layout.addWidget(edit_button)
        buttons_inner_layout.addStretch()
        buttons_inner_layout.addWidget(delete_button)
        
        buttons_layout.addWidget(buttons_container)
        buttons_layout.addStretch()
        
        # Tambahkan bagian tombol ke layout utama
        layout.addLayout(buttons_layout)


class CameraList(QWidget):
    """Widget kontainer yang menampilkan daftar kamera"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
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
                background-color: #27272A;;
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
        
        # Tambahkan beberapa kamera contoh
        self.add_sample_cameras()
    
    def add_sample_cameras(self):
        """Menambahkan beberapa kamera contoh"""
        self.add_camera("Sample Example Camera", "000.000.000.000", True)
        self.add_camera("Sample Example Camera", "000.000.000.000", True)
        self.add_camera("Sample Example Camera", "000.000.000.000", True)
        self.add_camera("Sample Example Camera", "000.000.000.000", True)
    
    def add_camera(self, name="Default Camera", ip="0.0.0.0", is_online=True):
        """Menambahkan kamera ke daftar"""
        camera_item = CameraItem(name, ip, is_online)
        self.cameras_layout.addWidget(camera_item)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        # Main window setup
        self.setWindowTitle("EyeLog - Computer Vision Monitoring")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #09090B;")
        self.setWindowIcon(QIcon("app/pdu.png"))
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

        # Title icon and text  for header
        header_icon_label = QLabel()
        header_icon_label.setPixmap(QIcon("app/assets/icons/pdu.png").pixmap(QSize(30, 30)))
        header_title_label = QLabel("PDU | Parama Data Unit")
        header_title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Clock widgets icon and text
        clock_icon_label = QLabel()
        clock_icon_label.setPixmap(QIcon("app/assets/icons/clock.png").pixmap(QSize(18, 18)))
        clock_text_label = QLabel("23:59:59 10-11-2024")
        clock_text_label.setStyleSheet("font-size: 14px")

        # Notification button icon
        notification_icon_label = QPushButton()
        notification_icon_label.setFixedSize(35, 35)
        notification_icon_label.setIcon(QIcon("app/assets/icons/bell.png"))
        notification_icon_label.setIconSize(QSize(20, 20))
        notification_icon_label.setStyleSheet("""
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
        notification_icon_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Add widgets to header
        header_layout.addWidget(header_icon_label)
        header_layout.addSpacing(4)
        header_layout.addWidget(header_title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(clock_icon_label)
        header_layout.addSpacing(3)
        header_layout.addWidget(clock_text_label)
        header_layout.addStretch(1)
        header_layout.addWidget(notification_icon_label)

        navbar_frame = QFrame()
        navbar_frame.setFrameShape(QFrame.Panel)
        navbar_frame.setFixedHeight(55)
        navbar_frame.setStyleSheet(""" 
            QFrame {background-color: #09090B; border: 0px solid transparent; border-bottom: 4px solid #EA580C;} 
        """)

        navbar_layout = QHBoxLayout(navbar_frame)
        navbar_layout.setContentsMargins(30, 0, 30, 0)


        navbar_title_label = QLabel("Computer Vision Video Realtime")
        navbar_title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: white; border: 0px;")

        navbar_add_camera = QPushButton("Add Camera")
        navbar_add_camera.setFixedHeight(35)
        navbar_add_camera.setStyleSheet("""
            QPushButton {
                background-color: #EA580C;
                color: #E4E4E7;
                font-size: 14px;
                border-radius: 8px;
                padding: 6px 12px;            }
            QPushButton:hover {
                background-color: #09090B;
                border: 1px solid #EA580C;
            }
        """)
        navbar_add_camera.setIcon(QIcon("app/assets/icons/plus.png"))
        navbar_add_camera.setIconSize(QSize(18, 18))

        navbar_layout.addWidget(navbar_title_label)
        navbar_layout.addStretch(1)
        navbar_layout.addWidget(navbar_add_camera)
        
        # # Set scroll area widget
        self.camera_list = CameraList()
        # scroll_area.setWidget(self.camera_list)
        
        # Add widgets to main layout
        main_layout.addWidget(header_frame)
        main_layout.addWidget(navbar_frame)
        # main_layout.addWidget(scroll_area)
        main_layout.addWidget(self.camera_list)
        
        # Status bar
        self.statusBar().showMessage("4 Cameras Connected | Database: Connected")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #27272A;
                color: #E4E4E7;
            }
        """)


if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec())