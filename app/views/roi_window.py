import sys
import os
import cv2 as cv
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
from PySide6.QtCore import Qt, QSize, QTimer, QEvent
from PySide6.QtGui import QIcon, QPixmap, QImage

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(current_dir, '../utils'))
sys.path.append(utils_dir)


from ..resources import resource_path
from ..utils.roi_selector import ROISelector

class ROISelectorDialog(QDialog):
    """
    Dialog for ROI selection with integrated OpenCV functionality.
    """
    def __init__(self, frame=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Select ROI")
        self.setWindowIcon(QIcon(resource_path("assets/icons/crop.png")))
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #09090B;
                border: 1px solid #27272A;
            }
            QLabel {
                color: #E4E4E7;
                border: 0px;
            }
            QFrame {
                border: 0px;
            }
        """)

        self.frame = None
        self.roi_selector = None
        self.roi_points = None
        self.roi_image = None
        self.pixmap_scale_x = 1.0
        self.pixmap_scale_y = 1.0
        self.pixmap_offset_x = 0.0
        self.pixmap_offset_y = 0.0

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header_frame = QFrame()
        header_frame.setFixedHeight(57)
        header_frame.setStyleSheet("background-color: #EA580C; border: 0px;")

        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 0, 20, 0)

        header_title = QLabel("Select Region of Interest (ROI)")
        header_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")

        close_button = QPushButton()
        close_button.setFixedSize(35, 35)
        close_button.setIcon(QIcon("app/assets/icons/close.png"))
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

        main_layout.addWidget(header_frame)

        content_frame = QFrame()
        content_frame.setStyleSheet("border: 0px;")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        instructions_frame = QFrame()
        instructions_frame.setStyleSheet("background-color: #18181B;")
        instructions_frame.setFixedHeight(40)
        instructions_layout = QHBoxLayout(instructions_frame)
        instructions_layout.setContentsMargins(20, 0, 20, 0)

        instructions_label = QLabel("Klik untuk memilih titik. Buat persegi panjang dengan memilih 3 titik. Titik keempat akan ditentukan otomatis.")
        instructions_label.setStyleSheet("color: #A1A1AA; font-size: 13px;")
        instructions_layout.addWidget(instructions_label)

        content_layout.addWidget(instructions_frame)

        self.preview_frame = QFrame()
        self.preview_frame.setStyleSheet("background-color: #000000;")
        self.preview_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        preview_layout = QVBoxLayout(self.preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setMouseTracking(True)
        self.preview_label.installEventFilter(self)

        preview_layout.addWidget(self.preview_label)

        content_layout.addWidget(self.preview_frame, 1)

        main_layout.addWidget(content_frame, 1)

        button_frame = QFrame()
        button_frame.setFixedHeight(70)
        button_frame.setStyleSheet("background-color: #27272A; border: 0px;")

        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(20, 0, 20, 0)
        button_layout.setSpacing(10)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedHeight(35)
        self.reset_button.setIcon(QIcon("app/assets/icons/refresh.png"))
        self.reset_button.setIconSize(QSize(16, 16))
        self.reset_button.setStyleSheet("""
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
        self.reset_button.clicked.connect(self.reset_selection)

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

        self.save_button = QPushButton("Save ROI")
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
        self.save_button.setIcon(QIcon("app/assets/icons/save.png"))
        self.save_button.setIconSize(QSize(16, 16))
        self.save_button.clicked.connect(self.confirm_selection)

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)

        main_layout.addWidget(button_frame)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)

        if frame is not None:
            self.set_frame(frame)

    def set_frame(self, frame):
        """Set the initial frame and initialize ROI selector."""
        if frame is not None:
            self.frame = frame.copy()

            self.roi_selector = ROISelector("dummy_name", self.frame, use_opencv_window=False)

            self.roi_selector.set_callback(self.update_from_selector)

            self.update_preview()

            self.timer.start(30)

    def update_from_selector(self, image):
        """Callback function called by ROISelector when its state changes."""
        pass

    def update_preview(self):
        """Update the preview with the current ROI selector state."""
        if self.roi_selector:
            display_image = self.roi_selector._update_display()

            if display_image is not None:
                rgb_image = cv.cvtColor(display_image, cv.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(qt_image)
                self.update_label_pixmap(pixmap)

    def update_label_pixmap(self, pixmap):
        """Update the preview label with the given pixmap, scaled appropriately."""
        if pixmap:
            label_size = self.preview_label.size()
            scaled_pixmap = pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.preview_label.setPixmap(scaled_pixmap)

            self.pixmap_scale_x = pixmap.width() / scaled_pixmap.width()
            self.pixmap_scale_y = pixmap.height() / scaled_pixmap.height()

            self.pixmap_offset_x = (label_size.width() - scaled_pixmap.width()) / 2
            self.pixmap_offset_y = (label_size.height() - scaled_pixmap.height()) / 2

    def reset_selection(self):
        """Reset the ROI selection."""
        if self.roi_selector:
            self.roi_selector.reset()

    def confirm_selection(self):
        """Confirm the current ROI selection."""
        if self.roi_selector and self.roi_selector.is_complete():
            self.roi_selector.confirm_selection()
            self.roi_points, self.roi_image = self.roi_selector.get_roi()
            if self.roi_points and self.roi_image is not None:
                self.accept()

    def get_roi(self):
        """Return the selected ROI points and image."""
        return self.roi_points, self.roi_image

    def eventFilter(self, obj, event):
        """Filter events for the preview label to handle mouse interactions."""
        if obj is self.preview_label:
            if event.type() == QEvent.Type.MouseButtonPress:
                pos = event.pos()
                if self.roi_selector and self.frame is not None:
                    x_rel = (pos.x() - self.pixmap_offset_x) * self.pixmap_scale_x
                    y_rel = (pos.y() - self.pixmap_offset_y) * self.pixmap_scale_y

                    if 0 <= x_rel < self.frame.shape[1] and 0 <= y_rel < self.frame.shape[0]:
                        if event.button() == Qt.MouseButton.LeftButton:
                            self.roi_selector._handle_click(int(x_rel), int(y_rel))
                        elif event.button() == Qt.MouseButton.RightButton:
                            self.roi_selector.reset()
                return True
            elif event.type() == QEvent.Type.MouseMove:
                pos = event.pos()
                if self.roi_selector and not self.roi_selector.is_complete():
                    x_rel = (pos.x() - self.pixmap_offset_x) * self.pixmap_scale_x
                    y_rel = (pos.y() - self.pixmap_offset_y) * self.pixmap_scale_y

                    if 0 <= x_rel < self.frame.shape[1] and 0 <= y_rel < self.frame.shape[0]:
                        self.roi_selector.current_point = (int(x_rel), int(y_rel))
                return True
            elif event.type() == QEvent.Type.KeyPress:
                key = event.key()
                if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
                    self.confirm_selection()
                    return True
                elif key == Qt.Key.Key_Escape:
                    self.reject()
                    return True

        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        """Handle resize events to update the preview."""
        super().resizeEvent(event)

        self.update_preview()

    def closeEvent(self, event):
        """Clean up resources when dialog is closed."""
        self.timer.stop()
        super().closeEvent(event)

