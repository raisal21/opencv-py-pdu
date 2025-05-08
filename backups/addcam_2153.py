import sys
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFormLayout, 
    QGroupBox, QSpinBox, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap, QFont


class AddCameraDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set dialog properties
        self.setWindowTitle("Add Camera Form")
        self.setWindowIcon(QIcon("app/assets/icons/webcam.png"))  # Use your camera icon path
        self.setMinimumSize(500, 350)
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
        header_title = QLabel("Add Camera")
        header_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        
        # Close button
        close_button = QPushButton()
        close_button.setFixedSize(35, 35)
        close_button.setIcon(QIcon("app/assets/icons/close.png"))  # Use your close icon path
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
        form_layout.setSpacing(16)
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Form labels
        name_label = QLabel("Camera Name:")
        name_label.setStyleSheet("font-size: 12px;")
        ip_label = QLabel("Camera IP:")
        ip_label.setStyleSheet("font-size: 12px;")
        port_label = QLabel("Port (optional):")
        port_label.setStyleSheet("font-size: 12px;")
        
        # Form fields
        self.camera_name = QLineEdit()
        self.camera_name.setPlaceholderText("Enter camera name")
        self.camera_name.setMinimumHeight(36)
        
        self.camera_ip = QLineEdit()
        self.camera_ip.setPlaceholderText("Enter IP address (e.g. 192.168.1.100)")
        self.camera_ip.setMinimumHeight(36)
        
        self.camera_port = QSpinBox()
        self.camera_port.setRange(1, 65535)
        self.camera_port.setValue(8080)
        self.camera_port.setMinimumHeight(36)
        
        # Add fields to form
        form_layout.addRow(name_label, self.camera_name)
        form_layout.addRow(ip_label, self.camera_ip)
        form_layout.addRow(port_label, self.camera_port)
        
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
        
        self.next_button = QPushButton("Next")
        self.next_button.setFixedHeight(35)
        self.next_button.setStyleSheet("""
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
        self.next_button.setIcon(QIcon("app/assets/icons/arrow-right.png"))  # Use your arrow icon
        self.next_button.setIconSize(QSize(18, 18))
        self.next_button.clicked.connect(self.accept)
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.next_button)
        
        # Add button frame to main layout
        main_layout.addWidget(button_frame)
    
    def get_camera_data(self):
        """Return the entered camera data"""
        return {
            'name': self.camera_name.text(),
            'ip': self.camera_ip.text(),
            'port': self.camera_port.value()
        }


# Example usage
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    dialog = AddCameraDialog()
    
    if dialog.exec():
        # This code runs when "Next" is clicked
        camera_data = dialog.get_camera_data()
        print(f"Camera Name: {camera_data['name']}")
        print(f"Camera IP: {camera_data['ip']}")
        print(f"Camera Port: {camera_data['port']}")
    else:
        # This code runs when "Cancel" is clicked
        print("Dialog canceled")
    
    sys.exit(0)