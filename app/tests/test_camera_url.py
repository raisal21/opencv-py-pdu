"""
Unit‑test terpisah untuk Camera.build_stream_url
Menjaga pemisahan konteks antara database CRUD dan helper URL.
 Jalankan:
    pytest -q tests/test_camera_url.py
"""

from camera import Camera

# ---------------------------------------------------------------------------
# URL Builder tests
# ---------------------------------------------------------------------------

def test_build_stream_url_without_auth():
    cam = Camera(ip_address="10.1.1.2", port=8554, stream_path="live")
    assert cam.build_stream_url() == "rtsp://10.1.1.2:8554/live"

def test_build_stream_url_with_auth():
    cam = Camera(ip_address="10.1.1.2", port=8554,
                 username="admin", password="123", stream_path="video")
    assert cam.build_stream_url() == "rtsp://admin:123@10.1.1.2:8554/video"

def test_build_stream_url_custom_override():
    custom = "rtsp://192.168.5.5:554/custom/path"
    cam = Camera(custom_url=custom)
    assert cam.build_stream_url() == custom
