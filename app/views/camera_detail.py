import sys, subprocess, os
import random 
import datetime
import threading
import numpy as np
import json 
import logging
import gc
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QComboBox, QPushButton, 
                               QFrame, QSizePolicy, QGridLayout, QSpacerItem,
                               QMessageBox)
from PySide6.QtCore import Qt, QThread, QSize, QRect, QTimer, QPropertyAnimation, QDateTime, QPointF, QMarginsF, QMargins, Slot, QUrl, QThreadPool
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QLinearGradient, QCursor, QIcon, QDesktopServices
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
from ..resources import resource_path
from ..models.camera import Camera, convert_cv_to_pixmap

# Import material detector classes
from ..utils.material_detector import (
    BG_PRESETS, 
    CONTOUR_PRESETS
)
from ..utils.coverage_logger import CoverageLogger
from .roi_window import ROISelectorDialog
from ..utils.stream_worker import StreamWorker, StreamState
from ..utils.db_worker import DBWorker, DBSignals

logger = logging.getLogger(__name__) 


class VerticalBarGraph(QFrame):
    """Custom widget to display a vertical line graph for time-series data using QChart"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 220)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("background-color: #27272A; border-radius: 8px; border: 1px solid #27272A;")
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create chart
        self.chart = QChart()
        self.chart.setBackgroundBrush(QBrush(QColor("#27272A")))
        self.chart.setBackgroundRoundness(0)
        self.chart.legend().hide()
        self.chart.setMargins(QMargins(5, 5, 5, 5))
        self.chart.setTitle("")
        
        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        layout.addWidget(self.chart_view)
        
        # Create series
        self.series = QLineSeries()
        pen = QPen(QColor("#EA580C"))
        pen.setWidth(3)
        self.series.setPen(pen)
        
        # Sample data structure with timestamps and values
        current_time = datetime.datetime.now()
        self.data = [
            {"timestamp": current_time - datetime.timedelta(seconds=0), "value": 87},
            {"timestamp": current_time - datetime.timedelta(seconds=5), "value": 75},
            {"timestamp": current_time - datetime.timedelta(seconds=10), "value": 65},
            {"timestamp": current_time - datetime.timedelta(seconds=15), "value": 70},
            {"timestamp": current_time - datetime.timedelta(seconds=20), "value": 90}
        ]
        
        # Create percentage axis (x-axis)
        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, 100)
        self.axis_x.setLabelsColor(QColor("#E4E4E7"))
        self.axis_x.setGridLineColor(QColor("#E4E4E7").darker(150))
        self.axis_x.setMinorGridLineColor(QColor("#E4E4E7").darker(200))
        self.axis_x.setTitleText("Coverage %")
        self.axis_x.setTitleBrush(QBrush(QColor("#E4E4E7")))
        self.axis_x.setLabelFormat("%d%%")
        self.axis_x.setTickCount(5)
        self.axis_x.setLabelsFont(QFont("Arial", 8))
        
        # Create time axis (y-axis) using QDateTimeAxis
        self.axis_y = QDateTimeAxis()
        self.axis_y.setLabelsColor(QColor("#E4E4E7"))
        self.axis_y.setGridLineColor(QColor("#E4E4E7").darker(150))
        self.axis_y.setMinorGridLineColor(QColor("#E4E4E7").darker(200))
        self.axis_y.setTitleText("Time")
        self.axis_y.setTitleBrush(QBrush(QColor("#E4E4E7")))
        self.axis_y.setFormat("hh:mm:ss")
        self.axis_y.setLabelsFont(QFont("Arial", 8))
        
        # Add the axes to the chart
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        # Add the series to the chart
        self.chart.addSeries(self.series)
        
        # Attach the series to the axes
        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)
        
        # Update with initial data
        self.update_data(self.data)
    
    def update_data(self, data):
        """Update the chart with new data that includes timestamps"""
        self.data = data
        
        # Clear existing points
        self.series.clear()
        
        # Get min/max timestamps for y-axis range
        if len(self.data) == 0:
            return
            
        min_time = min(item["timestamp"] for item in self.data)
        max_time = max(item["timestamp"] for item in self.data)
        
        # Convert to QDateTime for the axis
        min_qtime = QDateTime()
        min_qtime.setSecsSinceEpoch(int(min_time.timestamp()))
        
        max_qtime = QDateTime()
        max_qtime.setSecsSinceEpoch(int(max_time.timestamp()))
        
        # Set y-axis range with a small buffer
        buffer_seconds = 2  # Add 2 seconds buffer on each end
        buffer_min = min_qtime.addSecs(-buffer_seconds)
        buffer_max = max_qtime.addSecs(buffer_seconds)
        self.axis_y.setRange(buffer_min, buffer_max)
        
        # Add points to series (x=percentage, y=timestamp)
        for item in self.data:
            x_value = item["value"]
            
            # Convert Python datetime to QDateTime
            y_time = QDateTime()
            y_time.setSecsSinceEpoch(int(item["timestamp"].timestamp()))
            
            # Create QPointF with x=value, y=timestamp in milliseconds since epoch
            self.series.append(QPointF(x_value, y_time.toMSecsSinceEpoch()))


class CoverageStatusWidget(QFrame):
    """Widget to display current coverage status with color indicator"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.percentage = 87  # Default percentage
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(80)
        self.setStyleSheet("border-radius: 8px; border: 1px solid #27272A;")
        
        # Layout
        layout = QHBoxLayout(self)
        
        # Percentage label
        self.percentage_label = QLabel(f"{self.percentage}%")
        self.percentage_label.setStyleSheet("font-size: 24px; font-weight: bold; background-color: transparent; color: #E4E4E7; border-radius: 8px;")
        self.percentage_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.percentage_label)        

        # Status indicator
        self.indicator = QFrame()
        self.indicator.setMinimumSize(QSize(30, 30))
        self.indicator.setMaximumSize(QSize(30, 30))
        self.indicator.setStyleSheet("background-color: #FFEB3B; border-radius: 15px;")
        layout.addWidget(self.indicator)
        
        # Set background color based on percentage
        self.updateStatus(self.percentage)
        
    def updateStatus(self, percentage):
        """Update the percentage and indicator color"""
        self.setStyleSheet("border-radius: 8px; border: 1px solid #27272A;")
        self.percentage = percentage
        self.percentage_label.setText(f"{percentage}%")
        
        # Update indicator color based on thresholds
        if percentage >= 95:
            self.indicator.setStyleSheet("background-color: #F44336; border-radius: 15px;")  # Red
            self.setStyleSheet("background-color: rgba(244, 67, 54, 0.1);")
        elif percentage >= 85:
            self.indicator.setStyleSheet("background-color: #FFEB3B; border-radius: 15px;")  # Yellow
            self.setStyleSheet("background-color: rgba(255, 235, 59, 0.1);")
        else:
            self.indicator.setStyleSheet("background-color: #4CAF50; border-radius: 15px;")  # Green
            self.setStyleSheet("background-color: rgba(76, 175, 80, 0.1);")


class VideoDisplayWidget(QFrame):
    """Custom widget for displaying video feed with 16:9 aspect ratio"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(560, 315)  # 16:9 ratio
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #27272A; border-radius: 5px; border: 1px solid #27272A;")
        
        # Placeholder text
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label for video feed
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("color: #E4E4E7; font-size: 16px;")
        self.video_label.setText("Waiting for video connection...")

        layout.addWidget(self.video_label)
    
    def update_frame(self, frame):
        """Update the video frame display"""
        if frame is not None:
            # convert OpenCV frame to QImage or QPixmap
            pixmap = convert_cv_to_pixmap(frame) 
        
            # Scale to fit while preserving aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )

            self.video_label.setPixmap(scaled_pixmap)
    
    def set_offline_message(self):
        """Set the label to show offline message"""
        self.video_label.setText("Camera is offline")
        self.video_label.setStyleSheet("color: #FF0000; font-size: 16px; font-weight: bold;")

    def set_connecting_message(self):
        """Set the label to show connecting message"""
        self.video_label.setText("Connecting to camera...")
        self.video_label.setStyleSheet("color: #FFA500; font-size: 16px; font-weight: bold;")
        

class CameraDetailUI(QMainWindow):
    """Main window for the Camera Detail interface"""
    
    def __init__(self, camera_data, parent=None):
        super().__init__(parent)

        self._is_closing = False
        self._cleanup_done = False

        # Save camera data
        self.camera_data = camera_data
        self.camera_instance = None

        self.worker_thread: QThread | None = None
        self.frame_processor: FrameProcessor | None = None
        
        # Current metrics storage - will be updated with frames but only displayed every 5 seconds
        self.current_coverage_percent = 0
        self.current_metrics = None
        
        # History data for graph with timestamps
        current_time = datetime.datetime.now()
        self.coverage_history = [
            {"timestamp": current_time - datetime.timedelta(seconds=0), "value": 87},
            {"timestamp": current_time - datetime.timedelta(seconds=5), "value": 75},
            {"timestamp": current_time - datetime.timedelta(seconds=10), "value": 65},
            {"timestamp": current_time - datetime.timedelta(seconds=15), "value": 70},
            {"timestamp": current_time - datetime.timedelta(seconds=20), "value": 90}
        ]

        self.setWindowTitle(f"Camera Detail - {camera_data['name']}")
        self.setWindowIcon(QIcon(resource_path("assets/icons/webcam.png")))
        self.resize(1000, 700)
        
        # Central widget and main layout
        central_widget = QWidget()
        central_widget.setStyleSheet("background-color: #09090B;")
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Header with title and buttons
        header_layout = QHBoxLayout()
        title_label = QLabel(f"Camera Detail - {camera_data['name']}")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #E4E4E7;")
        header_layout.addWidget(title_label)
        
        # Add spacer to push buttons to the right
        header_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Back button
        back_button = QPushButton("Back")
        back_button.setIcon(QIcon(resource_path("assets/icons/back.png")))
        back_button.setIconSize(QSize(18, 18))
        back_button.setStyleSheet("""
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
        back_button.setFixedHeight(35)
        back_button.clicked.connect(self.close)
        header_layout.addWidget(back_button)
        
        main_layout.addLayout(header_layout)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #EA580C;")
        main_layout.addWidget(separator)
        
        # Content area - split into left and right sides
        content_layout = QHBoxLayout()
        
        # Left side layout (video + detection controls)
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
        
        # Right side layout (status and graph)
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)

        # Edit ROI and Data Logs buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        
        self.edit_roi_button = QPushButton("Edit ROI")
        self.edit_roi_button.setIcon(QIcon(resource_path("assets/icons/edit.png")))
        self.edit_roi_button.setIconSize(QSize(18, 18))
        self.edit_roi_button.setMinimumHeight(35)
        self.edit_roi_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        self.edit_roi_button.clicked.connect(self.edit_roi)
        button_row.addWidget(self.edit_roi_button, 1)

        data_logs_button = QPushButton("Logs Folder")
        data_logs_button.setIcon(QIcon(resource_path("assets/icons/logs.png")))
        data_logs_button.setIconSize(QSize(18, 18))
        data_logs_button.setMaximumHeight(35)
        data_logs_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        data_logs_button.clicked.connect(self.show_data_logs)
        button_row.addWidget(data_logs_button, 1)        
        right_layout.addLayout(button_row)
        
        # Coverage Status Panel
        status_layout = QVBoxLayout()
        status_title = QLabel("COVERAGE STATUS")
        status_title.setAlignment(Qt.AlignTop)
        status_title.setFixedHeight(40)
        status_title.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #27272A; border-radius: 8px; padding: 5px;")
        status_layout.addWidget(status_title)
        
        self.coverage_status = CoverageStatusWidget()
        status_layout.addWidget(self.coverage_status)
        right_layout.addLayout(status_layout)
        right_layout.addSpacing(8)
        
        # Graph Panel
        graph_layout = QVBoxLayout()
        graph_title = QLabel("COVERAGE HISTORY")
        graph_title.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #27272A; border-radius: 8px; padding: 5px;")
        graph_layout.addWidget(graph_title)
        
        self.graph = VerticalBarGraph()
        graph_layout.addWidget(self.graph)
        right_layout.addLayout(graph_layout)
        
        content_layout.addLayout(right_layout, 35)
        
        main_layout.addLayout(content_layout)
        
        # Initialize camera and material detection
        self.init_camera()
        self.coverage_logger = CoverageLogger()
        
        # Setup UI update timer - moderate frequency
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.timeout.connect(self.update_coverage_display)
        self.ui_update_timer.start(5000)  # 5000 ms = 5 seconds
        
    
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
        """Update UI dari worker - dengan safety check"""
        if self._is_closing:
            return
            
        try:
            self.current_metrics = metrics or {}
            self.current_coverage_percent = int(
                metrics.get("processed_coverage_percent", 0)
            )
            self.video_display.update_frame(display_frame)
            self._last_frame = display_frame.copy()
        except RuntimeError:
            # Widget might be deleted
            pass
    
    def update_coverage_display(self):
        """Update coverage display with safety checks"""
        if self._is_closing:
            return
            
        try:
            if self.current_metrics:
                # Update the coverage status widget
                coverage_percent = int(self.current_coverage_percent)
                self.coverage_status.updateStatus(coverage_percent)
                
                # Add new data point to history with current timestamp
                current_time = datetime.datetime.now()
                new_data_point = {
                    "timestamp": current_time,
                    "value": coverage_percent
                }
                
                # Add to history and maintain only the most recent 5 entries
                self.coverage_history.insert(0, new_data_point)
                if len(self.coverage_history) > 5:
                    self.coverage_history = self.coverage_history[:5]
                
                # Update the graph with the new history
                self.graph.update_data(self.coverage_history)
                
                # Log measurement with thread-safe check
                if hasattr(self, 'coverage_logger') and self.coverage_logger:
                    self.coverage_logger.log_measurement(
                        self.camera_data['id'], 
                        coverage_percent
                    )
        except RuntimeError:
            # Widget might be deleted
            pass
            
    def on_connect_result(self, success: bool):
        if success:
            # Daftarkan handler frame setelah stream berjalan
            self.camera_instance.thread.frame_received.connect(self.handle_roi_frame)
            self.camera_instance.thread.error_occurred.connect(self.handle_camera_error)
            self.camera_instance.thread.connection_changed.connect(self.handle_connection_change)
        else:
            # Gagal connect
            self.video_display.set_offline_message()
            QMessageBox.warning(
                self,
                "Camera Connection Error",
                f"Failed to connect to camera: {self.camera_instance.last_error}"
            )
    
    def init_camera(self):
        """Initialize camera and start video stream"""
        # Create camera instance from the camera data
        self.camera_instance = Camera.from_dict(self.camera_data)
        
        if self.camera_data.get('roi_points'):
            self.camera_instance.set_roi(self.camera_data['roi_points'])

        self.video_display.set_connecting_message()
        self._init_camera_new()

    def _init_camera_new(self):
        """New StreamWorker implementation"""
        logger.info("Using new StreamWorker implementation")
        
        # Create worker thread
        bg_params = BG_PRESETS["default"]
        contour_params = CONTOUR_PRESETS["standard"]

        if hasattr(self, "stream_worker") and self.stream_worker and self.stream_worker.isRunning():
            return

        self.stream_worker = StreamWorker(
            self.camera_instance,
            bg_params,
            contour_params,
            parent=self
        )
        
        # Connect signals
        self.stream_worker.frame_ready.connect(self.on_processed_frame, Qt.QueuedConnection)
        self.stream_worker.error_occurred.connect(self.handle_camera_error, Qt.QueuedConnection)
        self.stream_worker.connection_changed.connect(self.handle_connection_change, Qt.QueuedConnection)
        self.stream_worker.state_changed.connect(self._on_state_changed, Qt.QueuedConnection)
        
        # Start worker
        self.stream_worker.start()
    
    @Slot(int)
    def _on_state_changed(self, state: int):
        """Handle StreamWorker state changes"""
        logger.info(f"Stream state changed: {StreamState(state).name}")
        
        if state == StreamState.RECONNECTING:
            self.video_display.set_connecting_message()
        elif state == StreamState.EDIT_ROI:
            # Disable certain UI elements during ROI edit
            self.bg_combo.setEnabled(False)
            self.contour_combo.setEnabled(False)
            if hasattr(self, 'edit_roi_button'):
                self.edit_roi_button.setEnabled(False)
        elif state == StreamState.UPDATING_PRESETS:
            self.bg_combo.setEnabled(False)
            self.contour_combo.setEnabled(False)
            if hasattr(self, 'edit_roi_button'):
                self.edit_roi_button.setEnabled(False)
        elif state in (StreamState.RUNNING, StreamState.PAUSED):
            QTimer.singleShot(1000, lambda: self.bg_combo.setEnabled(True))
            QTimer.singleShot(1000, lambda: self.contour_combo.setEnabled(True))
            if hasattr(self, 'edit_roi_button'):
                QTimer.singleShot(1000, lambda: self.edit_roi_button.setEnabled(True))

    def edit_roi(self):
        """Open ROI selector dialog and update camera ROI points"""
        if not self.camera_instance:
            QMessageBox.warning(self, "Camera Not Ready",
                                "Cannot edit ROI while camera is offline.")
            return

        current_frame = self.stream_worker.get_latest_raw_frame()
        if current_frame is None:
            QMessageBox.warning(self, "No Frame Available",
                                "Could not get current frame from camera.")
            return

        self.stream_worker.enter_roi_edit()
        
        dialog = ROISelectorDialog(current_frame, self)
        if not dialog.exec():
            self.stream_worker.resume()
            return

        roi_points, _ = dialog.get_roi()
        if not roi_points:
            self.stream_worker.resume()
            return

        # simpan di objek kamera
        self.camera_instance.set_roi(roi_points)
        self.camera_data['roi_points'] = roi_points

        # konversi ke JSON
        roi_points_json = json.dumps([list(p) for p in roi_points])

        if 'id' not in self.camera_data:
            QMessageBox.warning(self, "Camera ID Missing",
                                "Could not identify camera ID for database update.")
            self.stream_worker.resume()
            return

        self.db_signals = DBSignals()
        def on_db_finished(result):
            if self._is_closing:
                logger.info("DB operation finished, but window is closing. Skipping MessageBox.")
                return

            if result:
                 QMessageBox.information(self, "ROI Diperbarui", "Region of interest telah berhasil diperbarui di database.")
            else:
                 QMessageBox.warning(self, "Gagal Memperbarui", "Gagal memperbarui ROI di database. Lihat log untuk detail.")

        def on_db_error(error_message):
            if self._is_closing:
                logger.warning(f"DB Error occurred but window is closing: {error_message}")
                return

            QMessageBox.critical(self, "Kesalahan Database", f"Terjadi kesalahan saat mengakses database:\n{error_message}")
            logger.error(error_message)

        self.db_signals.finished.connect(on_db_finished)
        self.db_signals.error.connect(on_db_error)

        # 3. Buat worker dan berikan objek sinyal serta nama metode dan argumennya
        worker = DBWorker(self.db_signals, "update_roi_points", self.camera_data['id'], roi_points_json)
        QThreadPool.globalInstance().start(worker)

        self.stream_worker.restart_with_new_roi(roi_points)
    
    def show_data_logs(self):
        cam_id = self.camera_data.get('id')
        if not cam_id:
            QMessageBox.warning(self, "Missing ID", "Camera ID not found.")
            return

        folder_path = self.coverage_logger._get_camera_dir(cam_id)

        # ⇩ ganti blok sinkron ⇩
        def _open_folder():
            # Qt‑way dulu
            if QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path)):
                return
            # Fallback native
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.call(["open", folder_path])
            else:
                subprocess.call(["xdg-open", folder_path])

        # non‑blocking flush
        self.coverage_logger.flush_async(_open_folder)

    def handle_camera_error(self, msg):
        logger.warning(f"[CameraError] {msg}")
        self.video_display.set_connecting_message()

    def handle_connection_change(self, ok):
        if ok:
            self.video_display.set_connecting_message()
        else:
            self.video_display.set_offline_message()
    
    def closeEvent(self, event):
        """
        Improved cleanup dengan urutan yang benar untuk mencegah segfault.
        """
        # 1. Cek apakah proses penutupan sudah berjalan untuk menghindari rekursi
        if self._is_closing:
            event.ignore() # Abaikan event close tambahan
            return
            
        self._is_closing = True
        logger.info("Starting cleanup for CameraDetailUI...")
        
        # 2. Hentikan semua timer yang berjalan di thread UI
        self.ui_update_timer.stop()

        if hasattr(self, 'db_signals') and self.db_signals:
            try:
                self.db_signals.disconnect()
                logger.info("DB signals disconnected.")
            except (TypeError, RuntimeError):
                pass
        
        # 3. Hentikan worker thread dengan SANGAT aman
        if hasattr(self, 'stream_worker') and self.stream_worker:
            # 3a. Putuskan semua koneksi sinyal. Ini mencegah sinyal terakhir
            #     dikirim ke slot yang mungkin sudah tidak valid saat cleanup.
            try:
                self.stream_worker.frame_ready.disconnect()
                self.stream_worker.error_occurred.disconnect()
                self.stream_worker.connection_changed.disconnect()
                self.stream_worker.state_changed.disconnect()
                logger.info("StreamWorker signals disconnected.")
            except (TypeError, RuntimeError):
                # TypeError jika sinyal tidak punya koneksi, RuntimeError jika objek sudah rusak
                logger.warning("Could not disconnect signals, may already be disconnected.")

            # 3b. Minta worker untuk berhenti dan TUNGGU hingga selesai.
            if self.stream_worker.isRunning():
                logger.info("Stopping StreamWorker from CameraDetail...")
                self.stream_worker.stop()
                # Beri waktu maksimum 5 detik untuk berhenti dengan baik
                if not self.stream_worker.wait(5000): 
                    logger.warning("StreamWorker did not stop gracefully, terminating.")
                    self.stream_worker.terminate() # Langkah berisiko, tapi perlu untuk UI yg responsif
            
            # 3c. Jadwalkan penghapusan worker oleh event loop. Ini cara teraman.
            self.stream_worker.deleteLater()
            self.stream_worker = None # Hapus referensi internal
            logger.info("StreamWorker stopped and scheduled for deletion.")
            
        if hasattr(self, "coverage_logger") and self.coverage_logger:
            try:
                self.coverage_logger.flush()
                self.coverage_logger.stop()
                gc.collect()
            except Exception as e:
                logger.warning(f"Failed stopping CoverageLogger: {e}")

        # 4. Setelah semua thread berhenti, baru tampilkan kembali parent
        parent = self.parent()
        if parent:
            try:
                # Pastikan parent adalah widget yang valid sebelum ditampilkan
                from shiboken6 import isValid

                if isValid(parent):
                    parent.show()
            except RuntimeError:
                logger.warning("Parent widget was already deleted.")


        # Tandai cleanup selesai
        self._cleanup_done = True
        
        # 5. Terima event close, biarkan Qt melanjutkan penghancuran widget
        logger.info("CameraDetailUI cleanup complete. Accepting close event.")
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Sample camera data for testing
    sample_camera_data = {
        'name': 'Test Camera',
        'ip': '0',
        'port': 0
    }
    
    window = CameraDetailUI(sample_camera_data)
    window.show()
    sys.exit(app.exec())