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
        
        self.video_display = VideoDisplayWidget()
        left_layout.addWidget(self.video_display)
        
        detection_panel = QFrame()
        detection_panel.setStyleSheet("background-color: #27272A; border-radius: 8px; border: 1px solid #27272A;")
        detection_layout = QVBoxLayout(detection_panel)
        detection_title = QLabel("DETECTION MODE (DISABLED FOR STATIC DEMO)")
        detection_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #A1A1AA;")
        detection_layout.addWidget(detection_title)
        
        self.bg_combo = QComboBox()
        self.bg_combo.setEnabled(False)
        self.contour_combo = QComboBox()
        self.contour_combo.setEnabled(False)
        
        detection_layout.addWidget(QLabel("Background:"))
        detection_layout.addWidget(self.bg_combo)
        detection_layout.addWidget(QLabel("Contour:"))
        detection_layout.addWidget(self.contour_combo)
        
        left_layout.addWidget(detection_panel)
        content_layout.addLayout(left_layout, 65)
        
        # Right Side
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)

        button_row = QHBoxLayout()
        self.edit_roi_button = QPushButton("Edit ROI")
        self.edit_roi_button.setEnabled(False)
        self.edit_roi_button.setToolTip("Fitur tidak tersedia untuk kamera statis")
        self.edit_roi_button.setStyleSheet("background-color: #3F3F46; color: #71717A; border-radius: 5px;")
        
        data_logs_button = QPushButton("Logs Folder")
        data_logs_button.setEnabled(False)
        data_logs_button.setToolTip("Fitur tidak tersedia untuk kamera statis")
        data_logs_button.setStyleSheet("background-color: #3F3F46; color: #71717A; border-radius: 5px;")

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
