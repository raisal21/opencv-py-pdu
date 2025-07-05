# camera_static_patch.py
# Modular patch for static (dummy) cameras
# Place this in `app/utils/camera_static_patch.py` or similar

import cv2 as cv
from PySide6.QtCore import QSize
from ..models.camera import Camera, convert_cv_to_pixmap

# 1. Daftar static camera (bisa hardcode, json, dsb)
STATIC_CAMERAS = [
    {
        'id': -100,
        'name': 'Demo Video Camera',
        'ip_address': '',
        'port': 0,
        'protocol': 'FILE',
        'username': '',
        'password': '',
        'stream_path': r'C:/Users/PC-Windows/Documents/GitHub/opencv-py-pdu/resource/video_dir/ROI_01.mp4',
        'url': r'C:/Users/PC-Windows/Documents/GitHub/opencv-py-pdu/resource/video_dir/ROI_01.mp4',
        'custom_url': r'C:/Users/PC-Windows/Documents/GitHub/opencv-py-pdu/resource/video_dir/ROI_01.mp4',
        'is_online': True,
    },
    # Tambah camera lain di sini jika perlu
]

def get_static_cameras():
    """
    Return list of Camera objects (static only, not from DB)
    """
    result = []
    for cam in STATIC_CAMERAS:
        result.append(Camera.from_dict(cam))
    return result

def get_static_camera_by_id(camera_id):
    for cam in STATIC_CAMERAS:
        if cam['id'] == camera_id:
            return Camera.from_dict(cam)
    return None

def get_preview_for_static_camera(cam_dict, size=QSize(160, 90)):
    if cam_dict.get('protocol') == 'FILE' and cam_dict.get('url'):
        cap = cv.VideoCapture(cam_dict['url'])
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return convert_cv_to_pixmap(frame, size)
    return None

def is_static_camera_id(camera_id):
    # Ganti sesuai policy (misal: semua id < 0 dianggap static)
    return isinstance(camera_id, int) and camera_id < 0
