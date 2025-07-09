import sys, subprocess, os
import datetime
import numpy as np
import json 
import logging
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QComboBox, QPushButton, 
                               QFrame, QSizePolicy, QSpacerItem,
                               QMessageBox)
from PySide6.QtCore import Qt, QSize, QTimer, Slot, QUrl
from PySide6.QtGui import QIcon
from ..resources import resource_path
from ..models.camera import Camera
from ..utils.stream_worker_static import StreamWorkerStatic
from ..views.camera_detail import VideoDisplayWidget, VerticalBarGraph, CoverageStatusWidget
# Import material detector classes
from ..utils.material_detector_static import (
    BG_PRESETS, 
    CONTOUR_PRESETS
)

logger = logging.getLogger(__name__) 

class CameraDetailUI_Static(QMainWindow):
    """
    Jendela detail kamera yang disederhanakan untuk kamera statis (demo).
    Fitur edit ROI, logging, dan ganti preset dinonaktifkan.
    """
    
    def __init__(self, camera_data, parent=None):
        super().__init__(parent)

        self._is_closing = False
        self.camera_data = camera_data
        self.camera_instance = Camera.from_dict(camera_data)
        self.stream_worker = None
        
        self.current_coverage_percent = 0
        self.current_metrics = {}
        self.coverage_history = []

        self.setWindowTitle(f"Camera Detail (Static) - {camera_data['name']}")
        self.setWindowIcon(QIcon(resource_path("assets/icons/webcam.png")))
        self.resize(1000, 700)
        
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #09090B;")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel(f"Camera Detail - {camera_data['name']}")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #E4E4E7;")
        header_layout.addWidget(title_label)
        header_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        back_button = QPushButton("Back")
        back_button.setIcon(QIcon(resource_path("assets/icons/back.png")))
        back_button.setStyleSheet("""
            QPushButton { background-color: #EA580C; color: #E4E4E7; font-size: 14px; border-radius: 8px; padding: 6px 12px; }
            QPushButton:hover { background-color: #09090B; border: 1px solid #EA580C; }
        """)
        back_button.setFixedHeight(35)
        back_button.clicked.connect(self.close)
        header_layout.addWidget(back_button)
        main_layout.addLayout(header_layout)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #EA580C;")
        main_layout.addWidget(separator)
        
        # Content Area
        content_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        
        # Video display
        self.video_display = VideoDisplayWidget()
        left_layout.addWidget(self.video_display)
        left_layout.addSpacing(8)
        
        # Detection controls panel
        detection_panel = QFrame()
        detection_panel.setFrameShape(QFrame.StyledPanel)
        detection_panel.setMinimumHeight(150)
        detection_panel.setStyleSheet("background-color: #27272A; border-radius: 8px; border: 1px solid #27272A;")
        
        detection_layout = QVBoxLayout(detection_panel)
        detection_title = QLabel("DETECTION MODE")
        detection_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #E4E4E7;")
        detection_layout.addWidget(detection_title)
        
        # Background dropdown
        bg_layout = QVBoxLayout()
        bg_label = QLabel("Background:")
        bg_label.setStyleSheet("color: #E4E4E7;")
        bg_layout.addWidget(bg_label)
        
        self.bg_combo = QComboBox()
        self.bg_combo.setStyleSheet("color: #E4E4E7; border-radius: 8px; background-color: #09090B; padding: 5px;")
        self.bg_combo.setMinimumHeight(35)
        self.bg_combo.setMaximumWidth(200)

        self.bg_preset_keys = list(BG_PRESETS.keys())
        for preset_name in self.bg_preset_keys:
            display_name = preset_name.replace('-', ' ').title()
            self.bg_combo.addItem(display_name)
            
        self.bg_combo.currentIndexChanged.connect(self.update_background_preset)
        bg_layout.addWidget(self.bg_combo)
        detection_layout.addLayout(bg_layout)
        
        # Contour dropdown
        contour_layout = QVBoxLayout()
        contour_label = QLabel("Contour:")
        contour_label.setStyleSheet("color: #E4E4E7;")
        contour_layout.addWidget(contour_label)
        
        self.contour_combo = QComboBox()
        self.contour_combo.setStyleSheet("color: #E4E4E7; border-radius: 8px; background-color: #09090B; padding: 5px;")
        self.contour_combo.setMinimumHeight(35)
        self.contour_combo.setMaximumWidth(200)

        self.contour_preset_keys = list(CONTOUR_PRESETS.keys())
        for preset_name in self.contour_preset_keys:
            # Convert preset names to title case for better display
            display_name = preset_name.title()
            self.contour_combo.addItem(display_name)
            
        self.contour_combo.currentIndexChanged.connect(self.update_contour_preset)
        contour_layout.addWidget(self.contour_combo)

        detection_layout.addLayout(contour_layout)
        detection_layout.addSpacing(4)
        
        # Reset Background button
        button_layout = QHBoxLayout()        
        reset_button = QPushButton("Reset Background")
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #A1A1AA;
                color: white;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #EA580C;
            }
        """)
        
        button_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        detection_layout.addLayout(button_layout)
        
        left_layout.addWidget(detection_panel)
        content_layout.addLayout(left_layout, 65)
        content_layout.addSpacing(8)

        # Right Side
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)

        # Edit ROI and Data Logs buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        button_row = QHBoxLayout()
        self.edit_roi_button = QPushButton("Edit ROI")
        self.edit_roi_button.setIcon(QIcon(resource_path("app/assets/icons/edit.png")))
        self.edit_roi_button.setIconSize(QSize(18, 18))
        self.edit_roi_button.setMinimumHeight(35)
        self.edit_roi_button.setEnabled(False)
        self.edit_roi_button.setToolTip("Fitur tidak tersedia untuk kamera statis")
        self.edit_roi_button.setStyleSheet("""
            QPushButton {
                background-color: #A1A1AA;
                color: white;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #EA580C;
            }
        """)
        
        data_logs_button = QPushButton("Logs Folder")
        data_logs_button.setIcon(QIcon(resource_path("app/assets/icons/logs.png")))
        data_logs_button.setIconSize(QSize(18, 18))
        data_logs_button.setMaximumHeight(35)
        data_logs_button.setEnabled(False)
        data_logs_button.setToolTip("Fitur tidak tersedia untuk kamera statis")
        data_logs_button.setStyleSheet("""
            QPushButton {
                background-color: #A1A1AA;
                color: white;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #EA580C;
            }
        """)

        button_row.addWidget(self.edit_roi_button)
        button_row.addWidget(data_logs_button)
        right_layout.addLayout(button_row)
        
        self.coverage_status = CoverageStatusWidget()
        self.graph = VerticalBarGraph()
        right_layout.addWidget(QLabel("COVERAGE STATUS"))
        right_layout.addWidget(self.coverage_status)
        right_layout.addWidget(QLabel("COVERAGE HISTORY"))
        right_layout.addWidget(self.graph)
        
        content_layout.addLayout(right_layout, 35)
        main_layout.addLayout(content_layout)
        
        self.init_camera_stream()

        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_coverage_display)
        self.ui_update_timer.start(5000)

    def init_camera_stream(self):
        self.video_display.set_connecting_message()
        self.stream_worker = StreamWorkerStatic(self.camera_instance)
        self.stream_worker.frame_ready.connect(self.on_processed_frame)
        self.stream_worker.error_occurred.connect(self.handle_camera_error)
        self.stream_worker.start()

    def update_background_preset(self, index):
        """Update background subtraction parameters based on selected preset"""

        if index < 0 or index >= len(self.bg_preset_keys):
            return
        # Get the preset key from the index
        preset_key = self.bg_preset_keys[index]
        
        # Get the preset parameters
        preset_params = BG_PRESETS[preset_key]

        self.stream_worker.set_bg_params(preset_params)
            
        # Show confirmation message
        QMessageBox.information(
            self,
            "Background Preset Applied",
            f"Background preset '{self.bg_combo.currentText()}' has been applied."
        )
    
    def update_contour_preset(self, index):
        """Update contour processing parameters based on selected preset"""
        if index < 0 or index >= len(self.contour_preset_keys):
            return
            
        # Get the preset key from the index
        preset_key = self.contour_preset_keys[index]
        
        # Get the preset parameters
        preset_params = CONTOUR_PRESETS[preset_key]

        self.stream_worker.set_contour_params(preset_params)
            
        # Show confirmation message
        QMessageBox.information(
            self,
            "Contour Preset Applied",
            f"Contour preset '{self.contour_combo.currentText()}' has been applied."
        )

    @Slot(np.ndarray, dict)
    def on_processed_frame(self, display_frame, metrics):
        if self._is_closing: return
        self.current_metrics = metrics or {}
        self.current_coverage_percent = int(metrics.get("processed_coverage_percent", 0))
        self.video_display.update_frame(display_frame)

    def update_coverage_display(self):
        if self._is_closing: return
        coverage_percent = self.current_coverage_percent
        self.coverage_status.updateStatus(coverage_percent)
        
        new_data_point = {"timestamp": datetime.datetime.now(), "value": coverage_percent}
        self.coverage_history.insert(0, new_data_point)
        self.coverage_history = self.coverage_history[:5]
        self.graph.update_data(self.coverage_history)

    def handle_camera_error(self, msg):
        logger.error(f"Static Stream Error: {msg}")
        self.video_display.set_offline_message()

    def closeEvent(self, event):
        if self._is_closing:
            event.ignore()
            return
            
        self._is_closing = True
        logger.info("Closing static camera detail window...")
        self.ui_update_timer.stop()
        
        if self.stream_worker and self.stream_worker.isRunning():
            self.stream_worker.stop()
            if not self.stream_worker.wait(3000):
                self.stream_worker.terminate()
        
        parent = self.parent()
        if parent:
            parent.show()
            
        event.accept()
