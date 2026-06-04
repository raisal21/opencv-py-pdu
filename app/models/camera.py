import cv2 as cv
import numpy as np
import json
import logging
from PySide6.QtCore import Qt, QMutex, QMutexLocker
from PySide6.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)


class Camera:
    """RTSP camera configuration with cached ROI transform state."""
    def __init__(self, camera_id=None, name="Untitled Camera", ip_address="0.0.0.0", port=554,
                 username="", password="", stream_path="stream1", custom_url=None):
        """Create a camera configuration object."""
        self.id = camera_id
        self.name = name
        self.ip_address = ip_address
        self.port = port

        self.username = username
        self.password = password
        self.stream_path = stream_path
        self.custom_url = custom_url


        self.resolution = (640, 480)
        self.roi_points = None

        self._roi_M:      np.ndarray | None = None
        self._roi_size:   tuple[int, int] | None = None
        self._roi_mutex = QMutex()


    def build_stream_url(self):
        """Build the RTSP stream URL, preferring a custom URL when provided."""
        if self.custom_url and self.custom_url.strip():
            return self.custom_url.strip()

        auth_part = ""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"

        stream_path = self.stream_path if self.stream_path else "stream1"
        if stream_path.startswith('/'):
            stream_path = stream_path[1:]

        return f"rtsp://{auth_part}{self.ip_address}:{self.port}/{stream_path}"


    def _prepare_roi_transform(self, frame: np.ndarray) -> None:
        """Cache the perspective transform for the current ROI."""
        if self.roi_points is None or len(self.roi_points) < 4:
            return

        src_pts = np.array(self.roi_points, dtype=np.float32)

        w1, w2 = np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[2] - src_pts[3])
        h1, h2 = np.linalg.norm(src_pts[1] - src_pts[2]), np.linalg.norm(src_pts[3] - src_pts[0])
        width, height = int(max(w1, w2)), int(max(h1, h2))

        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32
        )

        self._roi_M = cv.getPerspectiveTransform(src_pts, dst_pts)
        self._roi_size = (width, height)

    def process_frame_with_roi(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or self.roi_points is None or len(self.roi_points) < 4:
            return frame

        with QMutexLocker(self._roi_mutex):
            if self._roi_M is None:
                self._prepare_roi_transform(frame)

            if self._roi_M is None or self._roi_size is None:
                return frame

            M_copy = self._roi_M.copy()
            size_copy = self._roi_size

        return cv.warpPerspective(frame, M_copy, size_copy)

    def set_roi(self, roi_points):
        """Set ROI points from a list or JSON string and reset cached transforms."""
        if roi_points and isinstance(roi_points, str):
            try:
                self.roi_points = json.loads(roi_points)
            except:
                self.roi_points = None
        else:
            self.roi_points = roi_points

        with QMutexLocker(self._roi_mutex):
            self.roi_points = roi_points
            self._roi_M = None
            self._roi_size = None

    @staticmethod
    def from_dict(data):
        """Build a Camera instance from a database row dictionary."""
        camera = Camera(
            camera_id=data.get('id'),
            name=data.get('name'),
            ip_address=data.get('ip_address'),
            port=data.get('port', 554),
            username=data.get('username', ''),
            password=data.get('password', ''),
            stream_path=data.get('stream_path', 'stream1'),
            custom_url=data.get('url', '')
        )

        if 'resolution' in data:
            camera.resolution = data['resolution']
        elif 'resolution_width' in data and 'resolution_height' in data:
            camera.resolution = (data['resolution_width'], data['resolution_height'])

        if 'roi_points' in data and data['roi_points']:
            try:
                camera.roi_points = json.loads(data['roi_points'])
            except:
                pass

        return camera


def convert_cv_to_pixmap(cv_frame, target_size=None):
    """Convert an OpenCV BGR frame to a Qt pixmap."""
    if cv_frame is None:
        return QPixmap()

    rgb_frame = cv.cvtColor(cv_frame, cv.COLOR_BGR2RGB)

    h, w, ch = rgb_frame.shape
    bytes_per_line = ch * w
    image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(image)

    if target_size is not None:
        pixmap = pixmap.scaled(target_size,
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
        )

    return pixmap

